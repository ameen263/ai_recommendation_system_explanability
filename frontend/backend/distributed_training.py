import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("DistributedTrainingALS") \
        .config("spark.sql.warehouse.dir", "file:/tmp/spark-warehouse") \
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2") \
        .getOrCreate()

    try:
        # Derive the data file path relative to this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "data", "merged_data.csv")

        logger.info(f"Loading data from {data_path}...")

        # Check if file exists
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            return

        # Adjust 'sep' based on your CSV format (tab or comma)
        df = spark.read.csv(
            data_path,
            sep="\t",
            header=True,
            inferSchema=False
        )

        # Convert columns to appropriate types
        df = df.withColumn("user_id", col("user_id").cast("integer")) \
            .withColumn("movie_id", col("movie_id").cast("integer")) \
            .withColumn("rating", col("rating").cast("float"))

        # Split data
        train, test = df.randomSplit([0.8, 0.2], seed=42)

        # ALS model setup
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

        # Fit ALS model
        model = als.fit(train)

        # Evaluate with RMSE
        predictions = model.transform(test)
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions)
        logger.info(f"Distributed ALS model RMSE: {rmse:.4f}")

        # Directory creation
        model_save_path = os.path.join(script_dir, "models", "als_model")
        os.makedirs(model_save_path, exist_ok=True)

        # Corrected model_params (Fixed Getter Calls)
        model_params = {
            "rank": model.rank,
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

        # Saving user/item factors and params
        user_factors_path = os.path.join(model_save_path, "user_factors")
        item_factors_path = os.path.join(model_save_path, "item_factors")

        model.userFactors.write.mode("overwrite").format("parquet").save(f"file:{user_factors_path}")
        model.itemFactors.write.mode("overwrite").format("parquet").save(f"file:{item_factors_path}")

        params_path = os.path.join(model_save_path, "model_params.pkl")
        with open(params_path, 'wb') as f:
            pickle.dump(model_params, f)

        logger.info("Model saved successfully.")

    except Exception as e:
        logger.exception(f"An error occurred during distributed training: {e}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()