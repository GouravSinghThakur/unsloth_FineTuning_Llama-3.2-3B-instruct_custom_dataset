iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=5, placeholder="Enter your prompt here..."),
    outputs=gr.Textbox(lines=15, label="Model Response"),  # Increased output size
    title="Fine-tuned Llama-3.2-3B-Instruct Chatbot",
    description="Ask the model anything related to the fine-tuning dataset content!"
)
iface.launch()
