# Twitter Cleanup – AI-Powered Content Classification and Filtering

## 📌 Overview

Twitter Cleanup is a full-stack machine learning project designed to analyze text content and classify it using a trained Natural Language Processing (NLP) model.

The project combines multiple components:

- 🧠 A trained text classification model
- ⚙️ A Python backend API
- 🌐 A browser extension
- 📝 A custom text input interface for testing predictions
- 🔄 Communication between the browser extension and the backend

The main goal is to create a system capable of analyzing text content, sending it to a machine learning model, receiving a prediction, and using that prediction to determine how the content should be handled.

The system follows this general flow:

```text
┌─────────────────────┐
│   Twitter/X Feed    │
│   or Custom Input   │
└──────────┬──────────┘
           │
           │ Text
           ▼
┌─────────────────────┐
│ Browser Extension   │
│ / User Interface    │
└──────────┬──────────┘
           │
           │ HTTP Request
           ▼
┌─────────────────────┐
│    FastAPI Backend  │
└──────────┬──────────┘
           │
           │ Text
           ▼
┌─────────────────────┐
│ Tokenizer / NLP     │
│ Preprocessing       │
└──────────┬──────────┘
           │
           │ Token IDs
           ▼
┌─────────────────────┐
│ Trained Neural      │
│ Network Model       │
└──────────┬──────────┘
           │
           │ Logits
           ▼
┌─────────────────────┐
│ Prediction +        │
│ Confidence Score    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Filtering Decision  │
└─────────────────────┘
```

---

# 🏗️ System Architecture

The project is divided into three major layers:

1. **Browser / Frontend Layer**
2. **Backend API Layer**
3. **Machine Learning Layer**

Each layer has a specific responsibility.

```text
                    ┌─────────────────────┐
                    │      Twitter/X      │
                    │        Page         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Browser Extension   │
                    │ Content Script      │
                    └──────────┬──────────┘
                               │
                        Extracted Text
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FastAPI Server   │
                    │  /predict Endpoint  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Tokenizer      │
                    │  Text → Token IDs   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Neural Network    │
                    │  Text Classifier    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Prediction Result   │
                    │ + Confidence Score  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Browser Extension   │
                    │ Filtering Action    │
                    └─────────────────────┘
```

---

# 📁 Project Structure

```text
Twitter_cleanup/
│
├── backend/
│   │
│   ├── main.py
│   │   └── FastAPI backend and prediction endpoints
│   │
│   ├── requirements.txt
│   │   └── Python dependencies required by the backend
│   │
│   └── ...
│
├── extensions/
│   │
│   ├── manifest.json
│   │   └── Browser extension configuration
│   │
│   ├── content.js
│   │   └── Interacts with the Twitter/X page
│   │
│   └── ...
│
├── neural_network_20newsgroup/
│   │
│   ├── main.py
│   │   └── Model training logic
│   │
│   ├── test.py
│   │   └── Model testing and evaluation
│   │
│   ├── data/
│   │   └── Training/evaluation dataset
│   │
│   ├── results_bbc/
│   │   └── Training checkpoints and intermediate results
│   │
│   └── saved_model_bbc/
│       └── Final trained model files
│
├── .gitignore
│   └── Files and directories excluded from Git
│
└── README.md
```

---

# 🧠 Machine Learning Pipeline

The machine learning system receives raw text and converts it into a format that the neural network can understand.

The complete pipeline is:

```text
Raw Text
   │
   ▼
┌─────────────────────────┐
│       Tokenization      │
│                         │
│ "Hello world"           │
│           ↓             │
│ [101, 7592, 2088, 102]  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Attention Mask /        │
│ Input Preparation       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Transformer / Neural   │
│       Network           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│         Logits          │
│                         │
│ [2.1, 0.3, 4.8, 1.2]    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│         Softmax         │
│                         │
│ Converts scores into    │
│ probabilities           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Highest Probability     │
│        Category         │
└─────────────────────────┘
```

## 1. Raw Text Input

The system starts with text extracted from a tweet/post or entered manually by the user.

For example:

```text
The new technology announcement was released today.
```

At this stage, the model cannot directly process the text.

Neural networks work with numerical values, so the text must first be converted into numerical tokens.

---

## 2. Tokenization

The tokenizer converts the input text into numerical token IDs.

Conceptually:

```text
"The new technology announcement"

            ↓

[101, 1996, 2047, 2974, 4413, 102]
```

The exact token IDs depend on the tokenizer used by the trained model.

The tokenizer may also create:

- `input_ids`
- `attention_mask`
- Padding
- Truncation

These values ensure that the model receives text in the format expected during training.

---

## 3. Neural Network Prediction

The tokenized text is passed into the trained classification model.

The model processes the relationships and patterns between words and generates a set of output scores called **logits**.

For example:

```text
Category 0 → 1.2
Category 1 → 4.8
Category 2 → 0.9
Category 3 → 2.1
```

These values are not probabilities yet.

---

## 4. Softmax Conversion

The logits are converted into probabilities using the Softmax function.

Conceptually:

```text
Raw Scores

[1.2, 4.8, 0.9, 2.1]

        ↓

Softmax

        ↓

[0.02, 0.88, 0.01, 0.09]
```

The highest probability is selected as the predicted category.

In this example:

```text
Predicted Category: Category 1
Confidence: 88%
```

---

# ⚙️ Backend Architecture

The backend acts as the bridge between the browser extension and the machine learning model.

Its primary responsibilities are:

- Loading the trained model
- Loading the tokenizer
- Receiving text through an API
- Preparing the text for inference
- Running the model
- Processing the prediction
- Returning the result as JSON

The request flow is:

```text
Browser Extension
        │
        │ POST Request
        ▼
POST /predict
        │
        ▼
Validate Request
        │
        ▼
Tokenize Text
        │
        ▼
Run Model Inference
        │
        ▼
Calculate Prediction
        │
        ▼
Return JSON Response
```

---

# 🔌 API Communication

The browser extension communicates with the backend through an HTTP API.

A conceptual request looks like:

```json
{
    "text": "Example text to classify"
}
```

The backend processes this text and returns a response similar to:

```json
{
    "prediction": "predicted_category",
    "confidence": 0.92
}
```

The extension can then use this response to determine what action should be taken.

For example:

```text
IF prediction belongs to allowed categories
    → Keep content visible

IF prediction belongs to blocked categories
    → Hide or block content
```

---

# 🌐 Browser Extension

The browser extension allows the machine learning system to interact with content directly inside the browser.

Instead of manually copying and pasting every piece of text into a separate application, the extension can interact with the content displayed on the target website.

The extension is responsible for:

- Detecting relevant text content
- Extracting the text
- Sending text to the backend
- Receiving classification results
- Applying the filtering logic

## Extension Workflow

```text
Twitter/X Page Loads
        │
        ▼
Content Script Starts
        │
        ▼
Find Relevant Content
        │
        ▼
Extract Text
        │
        ▼
Send Text to Backend
        │
        ▼
Wait for Prediction
        │
        ▼
Receive Classification
        │
        ▼
Apply Filtering Rule
        │
        ├── Allowed → Display
        │
        └── Blocked → Hide/Filter
```

---

# 🧪 Custom Text Testing

The project also supports manual text testing.

This is useful because it allows the model to be tested independently of the Twitter/X page.

The workflow is:

```text
User Types Text
      │
      ▼
Submit
      │
      ▼
Backend API
      │
      ▼
Tokenizer
      │
      ▼
ML Model
      │
      ▼
Prediction
      │
      ▼
Display Result
```

This makes debugging significantly easier.

Instead of wondering whether an incorrect result is caused by:

- The extension
- Text extraction
- The API
- The backend
- The model

the model can first be tested directly using custom input.

---

# 🔄 End-to-End Request Lifecycle

Consider the following input:

```text
A user creates a post containing some text.
```

The complete system processes it as follows:

### Step 1 — Content Detection

The browser extension detects a piece of relevant text on the page.

### Step 2 — Text Extraction

The content script extracts the text from the page's DOM.

For example:

```text
"This is the text that needs to be classified."
```

### Step 3 — API Request

The extension sends the text to the backend.

```text
Extension
    ↓
POST /predict
    ↓
FastAPI Backend
```

### Step 4 — Tokenization

The backend passes the text to the tokenizer.

```text
Text
 ↓
Tokenizer
 ↓
Token IDs + Attention Mask
```

### Step 5 — Model Inference

The tokenized input is passed through the trained model.

The model produces output scores for the available categories.

### Step 6 — Prediction

The backend selects the category with the highest score.

Optionally, it calculates a confidence value.

### Step 7 — JSON Response

The backend sends the result back to the extension.

### Step 8 — Filtering Decision

The extension checks the returned category.

Depending on the filtering rules, it may:

- Leave the content unchanged
- Mark the content
- Hide the content
- Block the content

---

# 🛠️ Technology Stack

## Backend

### Python

Python is used for the backend and machine learning components.

### FastAPI

FastAPI provides the HTTP API used for communication between the extension and the model.

Responsibilities include:

```text
Receive Request
      ↓
Validate Data
      ↓
Run Prediction
      ↓
Return Response
```

### Uvicorn

Uvicorn runs the FastAPI application locally.

---

## Machine Learning

### PyTorch

PyTorch is used for running the neural network.

### Hugging Face Transformers

Transformers provides:

- Tokenizers
- Pre-trained model architectures
- Text classification utilities
- Model loading and inference support

---

## Browser Extension

The extension uses standard web technologies:

- HTML
- CSS
- JavaScript
- Browser Extension APIs

The content script interacts with the webpage while the backend handles the computationally intensive machine learning tasks.

---

# 🚀 Installation and Setup

## Prerequisites

Before running the project, make sure you have:

- Python 3.9 or newer
- pip
- A Chromium-based browser for the extension
- Git
- The trained model files

---

## 1. Clone the Repository

```bash
git clone https://github.com/shreyjauhari/Twitter_cleanup.git
cd Twitter_cleanup
```

---

## 2. Create a Python Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

Install the backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Depending on the project setup, additional machine learning dependencies may be required:

```bash
pip install torch transformers
```

---

# 🧠 Setting Up the Trained Model

The trained model is **not included in this repository**.

The model contains large files, including:

```text
model.safetensors
```

GitHub does not allow normal Git repositories to store individual files larger than 100 MB.

The model must therefore be placed manually in:

```text
neural_network_20newsgroup/saved_model_bbc/
```

After placing the required model files in this directory, the backend should be able to load the model according to its configured model path.

The expected structure is conceptually:

```text
neural_network_20newsgroup/
│
└── saved_model_bbc/
    │
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── ...
```

> The exact files depend on the model and tokenizer configuration.

---

# ▶️ Running the Backend

Navigate to the backend directory:

```bash
cd backend
```

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The server should start on a local address similar to:

```text
http://127.0.0.1:8000
```

The browser extension can then send requests to the prediction endpoint.

For example:

```text
POST http://127.0.0.1:8000/predict
```

---

# 🧩 Loading the Browser Extension

To load the extension:

1. Open your Chromium-based browser.
2. Navigate to the extensions page.
3. Enable **Developer Mode**.
4. Click **Load unpacked**.
5. Select the `extensions/` directory.
6. Open the target Twitter/X page.
7. Ensure that the backend server is running.

The extension should now be able to communicate with the local backend.

---

# 🧪 Testing the Machine Learning Model

The model can be tested independently using:

```bash
python neural_network_20newsgroup/test.py
```

Testing can be performed using:

- Known examples
- Custom text
- Evaluation datasets
- Manually entered text

Example workflow:

```text
Custom Input
      ↓
Tokenizer
      ↓
Model
      ↓
Prediction
      ↓
Compare Result
```

---

# 📊 Training and Evaluation

The training code and evaluation logic are located in:

```text
neural_network_20newsgroup/
```

The project separates several important stages:

```text
Dataset
   │
   ▼
Tokenization
   │
   ▼
Training
   │
   ▼
Checkpoint Generation
   │
   ▼
Evaluation
   │
   ▼
Final Model
```

During training, checkpoints may be created periodically.

These checkpoints are stored in:

```text
neural_network_20newsgroup/results_bbc/
```

They are excluded from Git because they can be large and are not required to run the final project when the final trained model is already available.

---

# 🚫 Files Excluded From Git

The following directories are intentionally excluded:

```text
neural_network_20newsgroup/data/
neural_network_20newsgroup/results_bbc/
neural_network_20newsgroup/saved_model_bbc/
```

## Why?

### Dataset

Datasets can be large and may be obtained separately.

### Training Results

Intermediate checkpoints are generated during training and can consume significant storage.

### Trained Model

The trained model contains a file larger than GitHub's standard 100 MB file limit.

---

# ⚠️ Important: Model Availability

Because the trained model is excluded from the repository, cloning the repository alone will **not immediately provide a fully working inference system**.

After cloning, the model must be obtained separately and placed in:

```text
neural_network_20newsgroup/saved_model_bbc/
```

Without the trained model, the backend will not be able to perform predictions.

Future versions may use:

- Git LFS
- Hugging Face Model Hub
- Cloud storage
- A downloadable release artifact

to simplify model distribution.

---

# 🐛 Troubleshooting

## `git push` fails because a file is too large

If GitHub reports:

```text
File exceeds GitHub's file size limit of 100 MB
```

Make sure the large file is included in `.gitignore`.

For example:

```gitignore
neural_network_20newsgroup/saved_model_bbc/
```

If the file was already committed, remove it from Git tracking:

```bash
git rm -r --cached neural_network_20newsgroup/saved_model_bbc/
```

Then amend or create a new commit before pushing again.

---

## Backend Cannot Connect to the Model

Check:

1. The model exists locally.
2. The model path in the backend is correct.
3. All required model files are present.
4. Required Python packages are installed.

---

## Extension Cannot Reach the Backend

Check:

1. The FastAPI server is running.
2. The API URL is correct.
3. The extension is pointing to the correct local address.
4. CORS configuration allows the request if required.

The backend should normally be reachable at:

```text
http://127.0.0.1:8000
```

---

# 🔮 Future Improvements

Possible improvements include:

- [ ] Improve model accuracy with more training data
- [ ] Add additional content categories
- [ ] Improve classification confidence handling
- [ ] Add configurable filtering thresholds
- [ ] Add user-defined blocked categories
- [ ] Process dynamically loaded content more efficiently
- [ ] Reduce duplicate API requests
- [ ] Add request caching
- [ ] Add batch classification
- [ ] Deploy the backend to the cloud
- [ ] Add authentication
- [ ] Create a dashboard for model statistics
- [ ] Add automated tests
- [ ] Add CI/CD
- [ ] Host the trained model externally
- [ ] Improve browser extension performance

---

# 🔐 Security Considerations

Sensitive information should never be committed to GitHub.

The following should normally be excluded:

```text
.env
API keys
Database passwords
Access tokens
Private credentials
```

A future version can include a `.env.example` file containing variable names without real secret values.

---

# 🤝 Contributing

Contributions are welcome.

To contribute:

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/new-feature
```

3. Make your changes.
4. Test the project.
5. Commit your changes.

```bash
git add .
git commit -m "Add new feature"
```

6. Push the branch.

```bash
git push origin feature/new-feature
```

7. Open a Pull Request.

---

# 👤 Author

**Shrey Jauhari**

---

# 📄 License

This project currently does not specify a license.

If you plan to make the project open source, consider adding an appropriate license such as:

- MIT License
- Apache License 2.0
- GPL

---

## ⭐ Summary

Twitter Cleanup demonstrates how multiple technologies can be combined into a complete machine learning application:

```text
Browser Content
      ↓
Browser Extension
      ↓
FastAPI Backend
      ↓
Tokenizer
      ↓
Transformer-Based Text Classifier
      ↓
Prediction + Confidence
      ↓
Filtering Decision
      ↓
Cleaner User Experience
```

The project is designed as an end-to-end demonstration of how a trained NLP model can move from experimentation and training into a real-world application with a backend API and browser integration.
