using System;
using System.IO;
using System.Threading.Tasks;
using IrisClustering.Model;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime.Api;

namespace IrisClustering.Model
{
    public class IrisData
    {
        [Column("0")]
        public float SepalLength;

        [Column("1")]
        public float SepalWidth;

        [Column("2")]
        public float PetalLength;

        [Column("3")]
        public float PetalWidth;
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
}

namespace IrisClustering
{

    /// <summary>
    /// 鉴于不知道每朵花属于哪个分组，应选择非监管式机器学习任务。 为将数据归入不同的组，并使同一组中的元素互相之间更为相似（与其他组中的元素相比），应使用聚类分析机器学习任务。
    /// </summary>
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");
        /// <summary>
        /// C#7.1以上版本
        /// </summary>
        /// <param name="args"></param>
        /// <returns></returns>
        static async Task Main(string[] args)
        {
            PredictionModel<IrisData, ClusterPrediction> model = Train();
            await model.WriteAsync(_modelPath);
            Console.ReadKey();
            TestModel(model);
            Console.ReadKey();
        }
        private static PredictionModel<IrisData, ClusterPrediction> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<IrisData>(separator: ','));
            pipeline.Add(new ColumnConcatenator(
                "Features",
                "SepalLength",
                "SepalWidth",
                "PetalLength",
                "PetalWidth"));
            pipeline.Add(new KMeansPlusPlusClusterer() { K = 3 });
            var model = pipeline.Train<IrisData, ClusterPrediction>();
            return model;
        }


       

        static void TestModel(PredictionModel<IrisData, ClusterPrediction> model)
        {
            var prediction = model.Predict(TestIrisData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
        }
    }

    static class TestIrisData
    {
        internal static readonly IrisData Setosa = new IrisData
        {
            SepalLength = 5.1f,
            SepalWidth = 3.5f,
            PetalLength = 1.4f,
            PetalWidth = 0.2f
        };
    }
}
