using Microsoft.ML.OnnxRuntime;
using System.IO;
using TensorStack.Common;

namespace RyzenAI.Example
{
    public static class Provider
    {
        public static ExecutionProvider CreateProvider(GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            return new ExecutionProvider("RyzenExecutionProvider", OrtMemoryInfo.DefaultInstance, configuration =>
            {
                var sessionOptions = new SessionOptions
                {
                    EnableCpuMemArena = true,
                    EnableMemoryPattern = true,
                    GraphOptimizationLevel = optimizationLevel
                };

                var modelPath = Path.GetDirectoryName(configuration.Path);
                var modelCache = Path.Combine(modelPath, ".cache");
                var modelName = Path.GetFileName(modelPath);
                if (modelName.Equals("unet"))
                    sessionOptions.AddSessionConfigEntry("model_name", "SD15_UNET");
                else if (modelName.Equals("vae_decoder"))
                    sessionOptions.AddSessionConfigEntry("model_name", "SD15_DECODER");

                sessionOptions.AddSessionConfigEntry("dd_cache", modelCache);
                sessionOptions.RegisterCustomOpLibrary("onnx_custom_ops.dll");
                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            });
        }
    }
}
