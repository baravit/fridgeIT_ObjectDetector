from flask import Flask, flash, request, redirect, session, render_template, send_from_directory
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

def plot_finetuned_results(im, prob=None, boxes=None):
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
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    original_path = session.get('uploaded_img_file_path', None)
    return f"<p>original:</br><img style='width: 200px;height: 200px' src='{original_path}'/></br>processed:</br><img style='width: 200px;height: 200px' src='data:image/png;base64,{data}'/></p>"


num_classes = 5
finetuned_classes = [
    'butter', 'cottage', 'milk', 'mustard', 'cream'
]

# Loading model
model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=num_classes)

# Loading checkpoint
checkpoint = torch.load('checkpoint.pth', map_location='cpu')

# del checkpoint["model"]["class_embed.weight"]
# del checkpoint["model"]["class_embed.bias"]
model.load_state_dict(checkpoint["model"], strict=False)

model.eval()

# Runing flask app
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'single'

# Test route
@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    img_name = './milk535.png'
    im = Image.open(img_name)
    
    outputs = run_worflow(im, model)
    
    prob, boxes = filter_bboxes_from_outputs(outputs, im=im)
    return plot_finetuned_results(im, prob, boxes)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
   if request.method == 'POST':
        file = request.files['file']
        # Extracting uploaded data file name
        img_filename = secure_filename(file.filename)
        # Upload file to database (defined uploaded folder in static path)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        im = Image.open(file)
            
        outputs = run_worflow(im, model)
        
        prob, boxes = filter_bboxes_from_outputs(outputs, im=im)
        return plot_finetuned_results(im, prob, boxes)

@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
       
        im = Image.open(file)
            
        outputs = run_worflow(im, model)
        
        prob, boxes = filter_bboxes_from_outputs(outputs, im=im)
        return plot_finetuned_results(im, prob, boxes)

    return '''
    <!doctype html>
    <title>FridgeIT</title>
    <h1>FridgeIT</h1>
    <h2>Upload new File</h2>
    <form method="POST" action="" enctype="multipart/form-data">
      <p><input type="file" name="file"></p>
      <p><input type="submit" value="Submit"></p>
    </form>
    {% if file %}
    <img src={{file}} />
    {% endif %}
    '''
