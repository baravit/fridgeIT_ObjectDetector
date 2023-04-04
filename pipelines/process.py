import torch
import io

# set root directory
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from model import run_worflow, filter_bboxes_from_outputs, plot_finetuned_results
from PIL import Image

from firebase_admin import credentials, initialize_app, storage, db

import matplotlib.pyplot as plt
import cv2
from matplotlib.figure import Figure

finetuned_classes = [
    'butter', 'cottage', 'milk', 'mustard', 'cream'
]

cred = credentials.Certificate("./FBserviceAccountKey.json")
initialize_app(cred, {'storageBucket': 'fridgeit-d17ae.appspot.com'})

file_name = 'cuttent.png'
source = "current_picture/cuttent.png"
dest = './uploads/{}'.format(file_name)

def load_model():
    # Loading model
    num_classes = 5
    model = torch.hub.load('facebookresearch/detr',
                        'detr_resnet50',
                        pretrained=False,
                        num_classes=num_classes)

    # Loading checkpoint
    checkpoint = torch.load('checkpoint.pth', map_location='cpu')

    # del checkpoint["model"]["class_embed.weight"]
    # del checkpoint["model"]["class_embed.bias"]
    model.load_state_dict(checkpoint["model"], strict=False)

    return model.eval()

def get_image_from_storage():
    bucket = storage.bucket()
    blob = bucket.blob(source)
    blob.download_to_filename(dest)
    img = Image.open(dest)
    return img
    # print(blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET'))

    # return blob

def crop_and_store_image(img, boxes, probs):
    counter = 0
    for ind, box in enumerate(boxes):
        prob = probs[ind]
        cl = prob.argmax()
        product_class = f'{finetuned_classes[cl]}: {prob[cl]:0.2f}'

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[0] + box[2])
        y2 = int(box[1] + box[3])
        roi = img[y1:y2, x1:x2]
        counter += 1


        crop_path = '{}@@@{}.png'.format(product_class, counter)
        cv2.imwrite('./uploads/{}'.format(crop_path), roi)

        # im = Image.open('./uploads/{}'.format(crop_path))
        
        # expiration_date_results = run_expiration_date_workflow(im, expiration_date_model)
        # crop_path += '@@@{}'.format(expiration_date_results)

        bucket = storage.bucket()
        blob = bucket.blob('cropped/{}'.format(crop_path))
        blob.upload_from_filename('./uploads/{}'.format(crop_path))

        # new_doc = {
        #     "product_link": "LINK_TO_STORAGE",
        #     "expiration_date": "EXPIRATION_DATE",
        # }

        # db.add_doc(new_doc)
        

def main():

    #TODO: delete all images from storage in folder cropped and related document

    #load image from storage current_picture/cuttent.png
    im = get_image_from_storage()
    
    # load model 
    model = load_model()

    # predict model on image
    outputs = run_worflow(im, model)
        
    # getting bboxes and probs
    prob, boxes = filter_bboxes_from_outputs(outputs, im=im)

    # cropping products
    cv_im = cv2.imread(dest)
    crop_and_store_image(cv_im, boxes, prob)

    # return plot_finetuned_results(im, prob, boxes)

if __name__ == '__main__':
    main()