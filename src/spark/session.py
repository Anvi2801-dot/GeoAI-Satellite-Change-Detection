import os
import sys
import platform
import subprocess

def setup_cross_platform_java():
    """Dynamically sets JAVA_HOME based on OS and system environment without hardcoding paths."""
    # 1. Respect user's existing valid JAVA_HOME if set
    current_java_home = os.environ.get("JAVA_HOME")
    if current_java_home and os.path.exists(os.path.join(current_java_home, "bin", "java")):
        return

    os_type = platform.system()

    # 2. macOS Automatic Detection
    if os_type == "Darwin":
        try:
            java_home = subprocess.check_output(["/usr/libexec/java_home"]).decode("utf-8").strip()
            os.environ["JAVA_HOME"] = java_home
            return
        except Exception:
            pass

    # 3. Linux Fallback Search Paths
    elif os_type == "Linux":
        common_linux_paths = [
            "/usr/lib/jvm/java-17-openjdk-amd64",
            "/usr/lib/jvm/default-java",
            "/usr/lib/jvm/java-11-openjdk-amd64"
        ]
        for path in common_linux_paths:
            if os.path.exists(os.path.join(path, "bin", "java")):
                os.environ["JAVA_HOME"] = path
                return

    # 4. Fallback: Search for java in system PATH
    try:
        java_executable = subprocess.check_output(["which" if os_type != "Windows" else "where", "java"]).decode("utf-8").strip().split('\n')[0]
        if java_executable:
            # Step up twice from bin/java to get the Java Home root
            os.environ["JAVA_HOME"] = os.path.dirname(os.path.dirname(java_executable))
    except Exception:
        pass

# Run automatic Java detection
setup_cross_platform_java()

# Fix macOS fork safety for PySpark / OpenMP
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# Prepend JAVA_HOME bin to system PATH if set
if "JAVA_HOME" in os.environ:
    os.environ["PATH"] = os.path.join(os.environ["JAVA_HOME"], "bin") + os.path.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from sedona.spark import SedonaContext

def init_geoai_spark_session(app_name: str = "GeoAI-Satellite-Change-Detection", driver_memory: str = "4g") -> SparkSession:
    
    # 2. Sedona Maven Coordinates matched specifically to Spark 3.4.x
    SEDONA_PACKAGES = (
        "org.apache.sedona:sedona-spark-shaded-3.4_2.12:1.5.1,"
        "org.datasyslab:geotools-wrapper:1.5.1-28.2"
    )

    # 3. Essential Java options for macOS & Java 17 execution
    java_opts = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
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