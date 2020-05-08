using System;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;
using System.Numerics;
using System.IO;
using System.Text;
using System.Linq;
using System.Collections.Generic;
// CMP304: Artificial Intelligence  - Lab 2 Example Code

namespace FeatureExtraction
{
    // The main program class
    class Program
    {

        //Gettting file names of files in a folder
        private static string[] NeutralImagesPaths = Directory.GetFiles("Neutral", "*.png").ToArray();
        private static string[] FearImagesPaths = Directory.GetFiles("Fear", "*.png").ToArray();
        private static string[] DisgustImagesPaths = Directory.GetFiles("Disgust", "*.png").ToArray();
        private static string[] AngerImagesPaths = Directory.GetFiles("Anger", "*.png").ToArray();
        private static List<string> Images;
        // The main program entry point
        static void Main(string[] args)
        {
            // header definition of the CSV file
            string header = "Label, LeftEyebrow, RightEyebrow, LeftLip, RightLip, LipWidth, LipHeight\n";
            System.IO.File.WriteAllText(@"feature_vectors.csv", header);
            Images = new List<string>();
            for (int i = 0; i < NeutralImagesPaths.Length; i++)
            {
                string temp = NeutralImagesPaths[i];
                Images.Add(temp);
            }
            for (int i = 0; i < FearImagesPaths.Length; i++)
            {
                string temp = FearImagesPaths[i];
                Images.Add(temp);
            }
            for (int i = 0; i < DisgustImagesPaths.Length; i++)
            {
                string temp = DisgustImagesPaths[i];
                Images.Add(temp);
            }
            for (int i = 0; i < AngerImagesPaths.Length; i++)
            {
                string temp = AngerImagesPaths[i];
                Images.Add(temp);
            }

            // Create Ouput Folders 
            string FolderPath = "NeutralOutput";
            if(!Directory.Exists(FolderPath))
            {
                Directory.CreateDirectory(FolderPath);
                Console.WriteLine("Created a new folder named: " + FolderPath);
            }

            // Create Ouput Folders 
            FolderPath = "AngerOutput";
            if(!Directory.Exists(FolderPath))
            {
                Directory.CreateDirectory(FolderPath);
                Console.WriteLine("Created a new folder named: " + FolderPath);
            }

            // Create Ouput Folders 
            FolderPath = "DisgustOutput";
            if(!Directory.Exists(FolderPath))
            {
                Directory.CreateDirectory(FolderPath);
                Console.WriteLine("Created a new folder named: " + FolderPath);
            }

            // Create Ouput Folders 
            FolderPath = "FearOutput";
            if(!Directory.Exists(FolderPath))
            {
                Directory.CreateDirectory(FolderPath);
                Console.WriteLine("Created a new folder named: " + FolderPath);
            }


            // SeinputFilePathst up Dlib Face Detector
            using (var fd = Dlib.GetFrontalFaceDetector())
            // ... and Dlib Shape DetectorS
            using (var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
            {
                int temp = 1;

                foreach (string str in Images)
                {
                    // load input image
                    var img = Dlib.LoadImage<RgbPixel>(str);

                    // find all faces i n the image
                    var faces = fd.Operator(img);
                    // for each face draw over the facial landmarks


                    // Create the CSV file and fill in the first line with the header
                    int shish = 0;
                    foreach (var face in faces)
                    {
                        // find the landmark points for this face
                        var shape = sp.Detect(img, face);

                        // draw the landmark points on the image
                        for (var i = 0; i < shape.Parts; i++)
                        {
                            var point = shape.GetPart((uint)i);
                            var rect = new Rectangle(point);
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);
                        }

                        /////////////// WEEK 9 LAB ////////////////

                        double[] LeftEyebrowDistances = new double[4];
                        double[] RightEyebrowDistances = new double[4];

                        float LeftEyebrowSum = 0;
                        float RightEyebrowSum = 0;

                        //LIP VARIABLES
                        double[] LeftLipDistances = new double[4];
                        double[] RightLipDistances = new double[4];
                        float LeftLipSum = 0;
                        float RightLipSum = 0;


                        LeftEyebrowDistances[0] = (shape.GetPart(21) - shape.GetPart(39)).Length;
                        LeftEyebrowDistances[1] = (shape.GetPart(20) - shape.GetPart(39)).Length;
                        LeftEyebrowDistances[2] = (shape.GetPart(19) - shape.GetPart(39)).Length;
                        LeftEyebrowDistances[3] = (shape.GetPart(18) - shape.GetPart(39)).Length;

                        RightEyebrowDistances[0] = (shape.GetPart(22) - shape.GetPart(42)).Length;
                        RightEyebrowDistances[1] = (shape.GetPart(23) - shape.GetPart(42)).Length;
                        RightEyebrowDistances[2] = (shape.GetPart(24) - shape.GetPart(42)).Length;
                        RightEyebrowDistances[3] = (shape.GetPart(25) - shape.GetPart(42)).Length;


                        //LIP
                        LeftLipDistances[0] = (shape.GetPart(51) - shape.GetPart(33)).Length;
                        LeftLipDistances[1] = (shape.GetPart(50) - shape.GetPart(33)).Length;
                        LeftLipDistances[2] = (shape.GetPart(49) - shape.GetPart(33)).Length;
                        LeftLipDistances[3] = (shape.GetPart(48) - shape.GetPart(33)).Length;

                            
                        RightLipDistances[0] = (shape.GetPart(51) - shape.GetPart(33)).Length;
                        RightLipDistances[1] = (shape.GetPart(52) - shape.GetPart(33)).Length;
                        RightLipDistances[2] = (shape.GetPart(53) - shape.GetPart(33)).Length;
                        RightLipDistances[3] = (shape.GetPart(54) - shape.GetPart(33)).Length;


                        for (int i = 0; i < 4; i++)
                        {
                            LeftEyebrowSum += (float)(LeftEyebrowDistances[i] / LeftEyebrowDistances[0]);
                            RightEyebrowSum += (float)(RightEyebrowDistances[i] / RightEyebrowDistances[0]);

                        }

                        LeftLipSum += (float)(LeftLipDistances[1] / LeftLipDistances[0]);
                        LeftLipSum += (float)(LeftLipDistances[2] / LeftLipDistances[0]);
                        LeftLipSum += (float)(LeftLipDistances[3] / LeftLipDistances[0]);


                        RightLipSum += (float)(RightLipDistances[1] / RightLipDistances[0]);
                        RightLipSum += (float)(RightLipDistances[2] / RightLipDistances[0]);
                        RightLipSum += (float)(RightLipDistances[3] / RightLipDistances[0]);

                        double LipWidth = (float)((shape.GetPart(48) - shape.GetPart(54)).Length / (shape.GetPart(33) - shape.GetPart(51)).Length);
                        double LipHeight = (float)((shape.GetPart(51) - shape.GetPart(57)).Length / (shape.GetPart(33) - shape.GetPart(51)).Length);

                        //compute label from parent's directory name
                        DirectoryInfo dr = new DirectoryInfo(str);
                        //Console.WriteLine(dr.Parent.Name.ToString());
                        string ParentFolderName = dr.Parent.Name.ToString();


                            

                        using (System.IO.StreamWriter file = new System.IO.StreamWriter(@"feature_vectors.csv", true))
                        {
                            file.WriteLine(ParentFolderName + "," + LeftEyebrowSum + "," + RightEyebrowSum + "," + LeftLipSum + "," + 
                                RightLipSum + "," + LipWidth + "," + LipHeight);
                        }

                        string filePath;
                        switch (ParentFolderName)
                        {
                            case "Neutral":
                                // export the modified image
                                filePath = "NeutralOutput/output" + temp.ToString() + ".jpg";
                                Dlib.SaveJpeg(img, filePath);
                                temp++;
                                break;

                            case "Anger":
                                // export the modified image
                                filePath = "AngerOutput/output" + temp.ToString() + ".jpg";
                                Dlib.SaveJpeg(img, filePath);
                                temp++;
                                break;

                            case "Disgust":
                                // export the modified image
                                filePath = "DisgustOutput/output" + temp.ToString() + ".jpg";
                                Dlib.SaveJpeg(img, filePath);
                                temp++;
                                break;

                            case "Fear":
                                // export the modified image
                                filePath = "FearOutput/output" + temp.ToString() + ".jpg";
                                Dlib.SaveJpeg(img, filePath);
                                temp++;
                                break;

                            default:
                                break;
                        }

                        //// export the modified image
                        //string filepath = "output" + ".jpg";
                        //Dlib.SaveJpeg(img, filepath);
                        //temp++;
                    }
                }//
            }
        }

    }
}