#--------------------------Import all pkgs and libraries------------

#Import all utils
import os
import sys
import io
import base64
import uuid
import re


#import numpy for image computation
import numpy as np

#import Image from PIL for image loading
from PIL import Image

#AWS SDK for Python3
import boto3
#Import json for all json related stuffs
import json

#our main web framework "Streamlit"
import streamlit as st 

#For streamlit designing and workaround
from utility import (
    img_to_bytes,
    read_markdown_file,
)

#-----------------All Imports are Completed----------------------

#-----------Configuring AWS-------------------------------------

DEFAULT_REGION = "us-east-2"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", DEFAULT_REGION)

#----------Assigning AWS Service------------
client = boto3.client("rekognition",region_name=AWS_DEFAULT_REGION,
aws_access_key_id=AWS_ACCESS_KEY_ID,
aws_secret_access_key=AWS_SECRET_ACCESS_KEY)





#Global variable
OBJECT_DEMO_IMAGE = "objimg.jpeg"
FACE_DEMO_IMAGE = "face.jpg"
DEMO_IMAGE = "demo.jpg"
PPE_DEMO_IMAGE = "ppe.jpg"
NUDE_DEMO_IMAGE = "nudity.jpg"



def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)  

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')  




#Download button function
def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. my.json,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    -------- 
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    object_to_download = json.dumps(object_to_download, ensure_ascii=True, indent=4)
    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: #ED732E;
                color: black;
                font-weight: 600;
                padding: 0.35em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 6px;
                border-width: 1px;
                border-style: solid;
                border-image: initial;
            }} 
            #{button_id}:hover {{
                color: white;
                border: none;
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
        custom_css
        + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'
    )

    return dl_link




#function to convert an image to bytes array
def pil_image_to_byte_array(image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, "PNG")
    return imgByteArr.getvalue()


#Function to perform the analysis
@st.cache
def process_image(image_bytes):
    response = client.detect_text(Image={"Bytes": image_bytes})
    return response

def object_detection(obj_img):
    response = client.detect_labels(Image={"Bytes": obj_img})
    return response

def face_analysis(face_img):
    response = client.detect_faces(Image={"Bytes": face_img}, Attributes=["ALL"])
    return response


def ppe_detection(ppe_image):
    response = client.detect_protective_equipment(Image={"Bytes": ppe_image},
    SummarizationAttributes={'MinConfidence':80, 'RequiredEquipmentTypes':
    ['FACE_COVER', 'HAND_COVER', 'HEAD_COVER']})
    return response
    
def nude_detection(nude_img):
    response = client.detect_moderation_labels(Image={"Bytes": nude_img})
    return response


#main function
def main():
    menu = ["Home","Object and Scene Detection",
    "Image Moderation",
    "Facial Analysis",
    "Text in Image",
    "PPE Detection",
    "About Me"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        about_md = read_markdown_file("markdowns/welcome.md")
        st.markdown(about_md, unsafe_allow_html=True)
    elif choice == "Object and Scene Detection":
        st.title("Detect objects and scenes in Image 🖼")
        object_file_buffer = st.file_uploader("Upload the image 👇", type = ["png", "jpg", "jpeg"])
        if object_file_buffer is not None:
            object_image_bytes = pil_image_to_byte_array(Image.open(object_file_buffer))
            object_image_array = np.array(Image.open(object_file_buffer))
        else:
            object_image_bytes = open(OBJECT_DEMO_IMAGE, "rb").read()
            object_image_array = np.array(Image.open(OBJECT_DEMO_IMAGE))

        st.write("Below is a default Image. You can upload your own custom image.👆")
        st.write(" ")
        st.image(object_image_array, use_column_width=True)
        st.write(" ")
        if st.button("Detect Labels"):
            response = object_detection(object_image_bytes)
            st.write(response)
            st.write(" ")
            download_button_str = download_button(response, "Labels.json", f"Click here to download the results")
            st.markdown(download_button_str, unsafe_allow_html=True)


    elif choice == "Facial Analysis":
        st.title("Analyze human faces 👦👩 in Image")
        face_file_buffer = st.file_uploader("Upload Image 👇", type= ["png", "jpg", "jpeg"])
        if face_file_buffer is not None:
            face_file_bytes = pil_image_to_byte_array(Image.open(face_file_buffer))
            face_file_array = np.array(Image.open(face_file_buffer))
        else:
            face_file_bytes = open(FACE_DEMO_IMAGE, "rb").read()
            face_file_array = np.array(Image.open(FACE_DEMO_IMAGE))

        st.write("Below is a default Image. You can upload your own custom image. 👆")
        st.write(" ")
        st.image(face_file_array, use_column_width=True)
        st.write(" ")

        if st.button("Analyze Face"):
            response = face_analysis(face_file_bytes)
            st.write(response)
            st.write(" ")
            download_button_str = download_button(response, "FaceAnalysis.json", f"Click here to download the results")
            st.markdown(download_button_str, unsafe_allow_html=True)


    elif choice == "PPE Detection":
        st.title("Detect If Humans have wore PPE 😷👷🏽")
        ppe_file_buffer = st.file_uploader("Upload Image Below 👇", type= ["png", "jpg", "jpeg"])
        if ppe_file_buffer is not None:
            ppe_file_bytes = pil_image_to_byte_array(Image.open(ppe_file_buffer))
            ppe_file_array = np.array(Image.open(ppe_file_buffer))
        else:
            ppe_file_bytes = open(PPE_DEMO_IMAGE, "rb").read()
            ppe_file_array = np.array(Image.open(PPE_DEMO_IMAGE))

        st.write("Below is a default Image. You can upload your own custom image. 👆")
        st.write(" ")
        st.image(ppe_file_array, use_column_width=True)
        st.write(" ")

        if st.button("Detect PPE"):
            response = ppe_detection(ppe_file_bytes)
            st.write(response)
            st.write(" ")
            download_button_str = download_button(response, "PPEresult.json", f"Click here to download the results")
            st.markdown(download_button_str, unsafe_allow_html=True)

    elif choice == "Image Moderation":
        st.title("Find Explicit or Nude Contents 🔞❌ ")
        nude_file_buffer = st.file_uploader("Upload an Image below 👇", type= ["png", "jpg", "jpeg"])
        if nude_file_buffer is not None:
            nude_file_bytes = pil_image_to_byte_array(Image.open(nude_file_buffer))
            nude_file_array = np.array(Image.open(nude_file_buffer))
        else:
            nude_file_bytes = open(NUDE_DEMO_IMAGE, "rb").read()
            nude_file_array = np.array(Image.open(NUDE_DEMO_IMAGE))

        st.write("Below is a default Image. You can upload your own custom image. 👆")
        st.write(" ")
        st.image(nude_file_array, use_column_width=True)
        st.write(" ")

        if st.button("Detect Explicit & Nude Contents"):
            response = nude_detection(nude_file_bytes)
            st.write(response)
            st.write(" ")
            download_button_str = download_button(response, "moderation.json", f"Click here to download the results")
            st.markdown(download_button_str, unsafe_allow_html=True)


    elif choice == "Text in Image":
        st.title("Extract Texts in Image 🅰️ 🅱️ ....")
        st.write(" ")
        st.write(" ")
        img_file_buffer = st.file_uploader("Upload an image 👇", type=["png", "jpg", "jpeg"])
        if img_file_buffer is not None:
            image_bytes = pil_image_to_byte_array(Image.open(img_file_buffer))
            image_array = np.array(Image.open(img_file_buffer))

        else:
            image_bytes = open(DEMO_IMAGE, "rb").read()
            image_array = np.array(Image.open(DEMO_IMAGE))
        st.write("Below is a default Image. You can upload your own custom image.👆")
        st.write(" ")
        st.image(image_array, use_column_width=True)
        st.write(" ")
        if st.button("Extract Texts"):
            response = process_image(image_bytes)
            extracted_text = [t["DetectedText"] for t in response["TextDetections"]]
            st.header("Extracted text")
            st.write(extracted_text)
            download_button_str = download_button(
            extracted_text, "extracted_text.json", f"Click here to download Extracted Text")
            st.write(" ")
            st.markdown(download_button_str, unsafe_allow_html=True)
            st.header("Raw response")
            st.write(response)
    else:
        about_markdown = read_markdown_file("markdowns/newabout.md")
        st.markdown(about_markdown, unsafe_allow_html=True)








#------------------------Ending the main function-----------------
if __name__ == '__main__':
    main()

#-----------------------App is completed---------------------------