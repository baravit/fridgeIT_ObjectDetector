import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import base64
from io import BytesIO
from werkzeug.utils import secure_filename
import os 


### Model vars and functions

finetuned_classes = [
    'butter', 'cottage', 'milk', 'mustard', 'cream'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def filter_bboxes_from_outputs(outputs,im,
                               threshold=0.8):
  
  # keep only predictions with confidence above threshold
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold

  prob = probas[keep]

  # convert boxes from [0; 1] to image scales
  boxes = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
  
  return prob, boxes

def run_worflow(my_image, my_model):
  # mean-std normalize the input image (batch-size: 1)
  img = transform(my_image).unsqueeze(0)

  # propagate through the model
  outputs = my_model(img)
  return outputs

def get_image_finetuned_results(im, prob=None, boxes=None):
    fig = Figure()
    ax = fig.gca()
    ax.axis('off')
    ax.imshow(im)
    colors = COLORS * 100
    
    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='black', alpha=0.5))
    
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    # data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return buf

def plot_finetuned_results(im, prob=None, boxes=None):
    plt.figure(figsize=(16,10))
    plt.imshow(im)
    ax = plt.gca()
    colors = COLORS * 100
    
    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='black', alpha=0.5))
    
    plt.axis('off')
    plt.show()




