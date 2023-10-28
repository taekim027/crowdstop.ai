import cv2
import sys
import os

def FrameCapture(vidpath, targetfolder):
    '''command line inputs as paths to video and target folder.
    function saves each frame as jpg in target folder'''

    if not os.path.exists(targetfolder):
        os.mkdir(targetfolder)


    vidObj = cv2.VideoCapture(vidpath)
    
    assert vidObj.isOpened(), "Video file cannot be opened."

    count = 1

    success, image = vidObj.read()

    success = 1

    while success:
        cv2.imwrite(targetfolder + f"/{count:06}.jpg", image)
        success, image = vidObj.read()

        count += 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python vid2img.py <video_file> <output_folder>")
    else:
        video_file = sys.argv[1]
        output_folder = sys.argv[2]
        FrameCapture(video_file, output_folder)
    