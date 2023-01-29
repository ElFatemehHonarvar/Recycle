# -*- coding: utf-8 -*-
"""bbx for github.ipynb

# Detection and Bounding Box
"""

# Commented out IPython magic to ensure Python compatibility.
# Command for running cli in google colab

%%writefile parsing.py

# Import Useful libraries 
import torch 
import argparse
import glob



def bounding_box_finder(img):
  """
  This function is designed to do the detection process and draw 
  a bounding box. Additionally, If the level of confidence for 
  the detected image is equal to or higher than the passed confidence, 
  the information about the box of the image will be displayed.


  Enter a image path and a number and a weight using cli  
  : param path : First input to bounding_box_finder
  : type path : str 
  : param confidence : Second input to bounding_box_finder
  : type confidence : int 
  : param weight : Third input to bounding_box_finder
  : type weight : str
  """

  count = 0    # number of detected image


# Detect object using yolov5 
  model = torch.hub.load('ultralytics/yolov5', 'custom', path=img.weight)
  all_imgs = glob.glob(img.path + '/*')
  results = model(all_imgs)
  results.save()


# Remove no detection and show the detected object if its confidence higher than or equal to the passing number
  for i in range(len(all_imgs)):
    if len(results.pandas().xyxy[i].index)==0:
      print(f"I am unable to draw bounding box for: {all_imgs[i]}")
      print("-----------------------")
      i+=1
    else:
      if results.pandas().xyxy[i].confidence.max()*100 >= img.confidence:
        count+=1  

        print(f"{results.pandas().xyxy[i].loc[results.pandas().xyxy[i].confidence.idxmax()]}")
        print(f"is result for {all_imgs[i]}")
        print(f"Model predict {count}  images from {len(all_imgs)}") 
        print("----------------------") 
        i+=1
        
          
      else:
        print(f"I am unable to draw bounding box for: {all_imgs[i]}  with choosen confidence")
        print("----------------------") 


# Prepare cli command
if __name__=="__main__":
  parser = argparse.ArgumentParser(
        description="welocome to the bounding box finder"
        )

  parser.add_argument(
      "-c","--confidence", 
      type=int,
      choices=range(0,100),
      help="Enter a number between 0 to 1"
    )

  parser.add_argument(
      "--path",
       type=str, 
       help="Enter the directory of a image"
       )

  parser.add_argument(
      "-w","--weight", 
      type=str,
      help="Enter weights of yolov5 training"
    )     



  parsed_args = parser.parse_args()


bounding_box_finder(parsed_args)

!python3 parsing.py -c <leve_of_confidence> --path <images_directory> -w <weights_directory>

