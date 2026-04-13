document.getElementById('send-btn').addEventListener('click', async () => {
    const userInput = document.getElementById('chat-input').value;
    if (!userInput) return;

    // Add user message to chat
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML += `<div class="user-message">${userInput}</div>`;

    // Clear the input field
    document.getElementById('chat-input').value = '';

    // Send the message to the backend
    try {
        const response = await fetch('http://localhost:5000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: userInput }),
        });

        const data = await response.json();

        if (data.reply) {
            // Add chatbot's reply to chat
            chatMessages.innerHTML += `<div class="bot-message">${data.reply}</div>`;
        } else {
            chatMessages.innerHTML += `<div class="bot-message">Error: ${data.error}</div>`;
        }

        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    } catch (error) {
        console.error('Error:', error);
        chatMessages.innerHTML += `<div class="bot-message">Error: Failed to connect to the chatbot.</div>`;
    }
});