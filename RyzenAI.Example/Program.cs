using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Image;
using TensorStack.Providers;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Pipelines.StableDiffusion;

namespace RyzenAI.Example
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            // Model Path
            var modelPath = "Models\\stable-diffusion-1.5-amdnpu";
            if (!Directory.Exists(modelPath))
            {
                Console.WriteLine("Enter Model Path:");
                modelPath = Console.ReadLine();
                if (!Directory.Exists(modelPath))
                {
                    Console.WriteLine("[Error] Model directory not found");
                    return;
                }
            }

            // Prompt
            Console.WriteLine("Enter Prompt:");
            var prompt = Console.ReadLine();
            if (string.IsNullOrEmpty(prompt))
            {
                Console.WriteLine("Empty prompt is invalid.");
                return;
            }

            // Generate
            await GenerateAsync(modelPath, prompt);
        }


        public static async Task GenerateAsync(string modelPath, string prompt)
        {
            var provider = Provider.GetProvider(DeviceType.GPU);
            var config = LoadConfig(modelPath, provider);
            using (var pipeline = new StableDiffusionPipeline(config))
            {
                var options = pipeline.DefaultOptions with { Prompt = prompt };
                var result = await pipeline.RunAsync(options);
                await result.SaveAsync("Output.png");
                ShowImageResult();
            }
        }


        private static StableDiffusionConfig LoadConfig(string modelPath, ExecutionProvider provider)
        {
            var config = StableDiffusionConfig.FromDefault("stable -diffusion-1.5-amdnpu", ModelType.Base, provider);
            config.Tokenizer.Path = Path.Combine(modelPath, "tokenizer", "vocab.json");
            config.TextEncoder.Path = Path.Combine(modelPath, "text_encoder", "model.onnx");
            config.Unet.Path = Path.Combine(modelPath, "unet", "model_NCHW.onnx");
            config.AutoEncoder.DecoderModelPath = Path.Combine(modelPath, "vae_decoder", "model_NCHW.onnx");
            return config;
        }


        private static void ShowImageResult()
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = "explorer.exe",
                Arguments = "Output.png",
                UseShellExecute = true
            });
        }
    }
}
