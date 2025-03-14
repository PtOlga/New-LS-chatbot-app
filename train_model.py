import json
import os
from datetime import datetime
from langchain_core.prompts import PromptTemplate

def analyze_chat_history():
    chat_history_dir = "chat_history"
    training_data = []
    
    # Collect all chat history files and analyze feedback
    for filename in os.listdir(chat_history_dir):
        if filename.endswith('.json'):
            with open(os.path.join(chat_history_dir, filename), 'r') as f:
                chats = json.load(f)
                # Filter chats that need improvement and have corrections
                for chat in chats:
                    if chat.get('feedback') == 'needs_improvement' and chat.get('corrections'):
                        training_data.append({
                            'context': chat['context'],
                            'question': chat['question'],
                            'original_answer': chat['answer'],
                            'improved_answer': chat['corrections']
                        })
    
    return training_data

def generate_improved_prompt(training_data):
    # Store pairs of original and improved answers for analysis
    improvements = []
    for item in training_data:
        improvements.append({
            'original': item['original_answer'],
            'improved': item['improved_answer']
        })
    
    # Generate enhanced prompt template based on collected feedback
    new_template = """
    You are a helpful legal assistant that answers questions based on information from status.law.
    Answer accurately and concisely.
    
    Based on previous feedback, please ensure your answers:
    - Are specific and detailed
    - Include relevant legal context
    - Provide practical next steps when applicable
    - Focus on international legal expertise
    - Maintain professional tone
    
    Question: {question}
    Context: {context}
    """
    
    return PromptTemplate.from_template(new_template)

def main():
    # Load and analyze training data
    training_data = analyze_chat_history()
    if training_data:
        improved_prompt = generate_improved_prompt(training_data)
        
        # Save new prompt template for future use
        with open('improved_prompt.txt', 'w') as f:
            f.write(str(improved_prompt))
            
        print(f"Analyzed {len(training_data)} feedback items")
        print("Generated improved prompt template")
    else:
        print("No feedback data available for analysis")

if __name__ == "__main__":
    main()