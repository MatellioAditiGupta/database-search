import json
#import boto3
from user_backend import *
from dotenv import load_dotenv
#import gradio as gr
print("Done")

load_dotenv('.env')

# def chatbot_interface(query):
#     try:
#         documents_all = load_documents("docs")
#         documents = split_documents(documents_all)
#         model = user_model(documents=documents)
#         chat_history = []

#         if query == '':
#             return ''

#         response = model({'question': query, 'chat_history': chat_history})
#         chat_history.append(response['answer'])
#         return response['answer'], process_metadata(response['source_documents'])

#     except Exception as e:
#         return f"Error in chatbot_interface: {str(e)}"

def lambda_handler(event, context):
    try:
        # Extract environment variables here if needed
        
        # Extract query from the event
        query = event.get("query", "")

        # Retrieve documents from S3
        documents_all = load_documents(bucket_name='mixdocs', s3_prefix='docs')

        # Process the documents and perform chatbot actions
        documents = split_documents(documents_all)
        model = user_model(documents=documents)
        chat_history = []

        if query == '':
            return {'statusCode': 200, 'body': ''}
        
        response = model({'question': query, 'chat_history': chat_history})
        chat_history.append(response['answer'])
        return {
            'statusCode': 200,
            'body': json.dumps({
                'answer': response['answer'],
                'metadata': process_metadata(response['source_documents'])
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

# iface = gr.Interface(fn=chatbot_interface, # Testing
#                      inputs="text", 
#                      outputs=["text","text"])

# iface.launch()