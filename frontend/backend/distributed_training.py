import os
import sys
import logging
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # Initialize Spark session with proper warehouse directory and configurations.
    spark = SparkSession.builder \
        .appName("DistributedTrainingALS") \
        .config("spark.sql.warehouse.dir", "file:/tmp/spark-warehouse") \
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2") \
        .getOrCreate()

    try:
        # Derive the data file path relative to this script's directory.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "data", "merged_data.csv")

        logger.info(f"Loading data from {data_path}...")
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            sys.exit(1)

        # Read CSV with header and schema inference enabled.
        df = spark.read.csv(
            data_path,
            sep="\t",
            header=True,
            inferSchema=True
        )

        # Explicitly cast columns to ensure correct types.
        df = df.withColumn("user_id", col("user_id").cast("integer")) \
               .withColumn("movie_id", col("movie_id").cast("integer")) \
               .withColumn("rating", col("rating").cast("float"))

        # Split data into training and testing sets.
        train, test = df.randomSplit([0.8, 0.2], seed=42)

        # Set up the ALS model with parameters.
        als = ALS(
            userCol="user_id",
            itemCol="movie_id",
            ratingCol="rating",
            nonnegative=True,
            implicitPrefs=False,
            coldStartStrategy="drop",
            rank=50,
            maxIter=10,
            regParam=0.1,
            seed=42
        )

        # Fit the ALS model on the training data.
        model = als.fit(train)

        # Evaluate the model using RMSE on the test set.
        predictions = model.transform(test)
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions)
        logger.info(f"Distributed ALS model RMSE: {rmse:.4f}")

        # Create a directory for saving the model if it doesn't exist.
        model_save_path = os.path.join(script_dir, "models", "als_model")
        os.makedirs(model_save_path, exist_ok=True)

        # Build a dictionary of model parameters using getter methods.
        model_params = {
            "rank": model.rank if hasattr(model, "rank") else als.getRank(),
            "maxIter": als.getMaxIter(),
            "regParam": als.getRegParam(),
            "userCol": als.getUserCol(),
            "itemCol": als.getItemCol(),
            "ratingCol": als.getRatingCol(),
            "nonnegative": als.getNonnegative(),
            "implicitPrefs": als.getImplicitPrefs(),
            "coldStartStrategy": als.getColdStartStrategy(),
            "rmse": rmse
        }

        # Save the user and item factors as Parquet files.
        user_factors_path = os.path.join(model_save_path, "user_factors")
        item_factors_path = os.path.join(model_save_path, "item_factors")

        model.userFactors.write.mode("overwrite").format("parquet").save(f"file:{user_factors_path}")
        model.itemFactors.write.mode("overwrite").format("parquet").save(f"file:{item_factors_path}")

        # Save the model parameters using pickle.
        params_path = os.path.join(model_save_path, "model_params.pkl")
        with open(params_path, 'wb') as f:
            pickle.dump(model_params, f)

        logger.info("Model saved successfully.")

    except Exception as e:
        logger.exception(f"An error occurred during distributed training: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
