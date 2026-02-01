Zero-Shot Image Classification Using CLIP For Interactive Kiosk Assistance

📌 Overview
This project demonstrates a zero-shot image classification system built using OpenAI’s CLIP model on the Animals with Attributes 2 (AWA2) dataset. The application is deployed using Streamlit and enhanced with LLaMA 3.3 70B Versatile to dynamically generate and answer context-aware questions about classified images.
The system classifies images without explicit training on every class, enabling recognition of both seen and unseen categories through semantic understanding.

📌 Features
-Zero-shot image classification using CLIP
-Supports unseen classes without labeled training data
-Attribute-based classification using a binary predicate matrix
-Cosine similarity for image–text matching
-Interactive Streamlit web application
-Dynamic Q&A powered by LLaMA 3.3 70B
-Performance comparison between ViT-B/16 and ViT-B/32

📌 How It Works
-User uploads an image through the Streamlit UI
-CLIP generates image embeddings
-Text prompts and class attributes are embedded
-Cosine similarity is used to match images with classes
-Attribute mappings enable zero-shot classification
-LLaMA 3.3 generates and answers questions based on the image context

📌  Dataset
-Animals with Attributes 2 (AWA2)
-50 animal classes
-85 semantic attributes
-Designed for zero-shot learning tasks

📌 Models Used
-CLIP
-ViT-B/16
-ViT-B/32
-Used to analyze the impact of transformer architecture on classification accuracy.
-LLaMA 3.3 70B Versatile

📌 Tech Stack
-Frontend: Streamlit
-Backend: Python, Flask
-ML Models: CLIP, LLaMA 3.3 70B
-Similarity Metric: Cosine Similarity

📌 Results
The system effectively identifies both seen and unseen image categories using zero-shot learning. Attribute-based classification combined with CLIP embeddings enables accurate predictions without class-specific training data. Transformer architecture comparisons show performance variation between ViT-B/16 and ViT-B/32 models.
