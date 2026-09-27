import os
import sys

# 1. Point explicitly to Java 11 Home
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from sedona.spark import SedonaContext

def init_geoai_spark_session(app_name: str = "GeoAI-Satellite-Change-Detection", driver_memory: str = "4g") -> SparkSession:
    
    # 2. Sedona Maven Coordinates matched specifically to Spark 3.4.x
    SEDONA_PACKAGES = (
        "org.apache.sedona:sedona-spark-shaded-3.4_2.12:1.5.1,"
        "org.datasyslab:geotools-wrapper:1.5.1-28.2"
    )

    # 3. Essential Java options for macOS execution
    java_opts = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
    )

    # 4. Construct PySpark Builder
    builder = SedonaContext.builder() \
        .appName(app_name) \
        .master("local[2]") \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.jars.packages", SEDONA_PACKAGES) \
        .config("spark.jars.excludes", "edu.ucar:cdm-core") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator") \
        .config("spark.sql.extensions", "org.apache.sedona.sql.SedonaSqlExtensions") \
        .config("spark.driver.extraJavaOptions", java_opts) \
        .config("spark.executor.extraJavaOptions", java_opts)

    spark = builder.getOrCreate()
    sedona = SedonaContext.create(spark)
    sedona.sparkContext.setLogLevel("WARN")

    print("[SUCCESS] Apache Sedona Spark Session Initialized successfully!")
    return sedona

if __name__ == "__main__":
    spark = init_geoai_spark_session()
    spark.sql("SELECT ST_Point(73.8567, 18.5204) AS pune_coords").show()
    spark.stop()
