// Nexon Chat Application
const API_BASE = window.location.origin;

// State
let messages = [];
let isGenerating = false;

// DOM Elements
const loadingState = document.getElementById('loading-state');
const emptyState = document.getElementById('empty-state');
const chatState = document.getElementById('chat-state');
const messagesEl = document.getElementById('messages');
const chatContainer = document.getElementById('chat-container');
const formEmpty = document.getElementById('chat-form-empty');
const formChat = document.getElementById('chat-form');
const inputEmpty = document.getElementById('input-empty');
const inputChat = document.getElementById('input');
const statsEl = document.getElementById('stats');

// Configure marked
marked.setOptions({ breaks: true, gfm: true });

// Check if model is ready
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        return data.ready;
    } catch {
        return false;
    }
}

// Wait for model to be ready
async function waitForModel() {
    const loadingText = loadingState.querySelector('.loading-text');
    let dots = 0;

    const updateText = () => {
        dots = (dots + 1) % 4;
        loadingText.textContent = 'Loading model' + '.'.repeat(dots);
    };

    const textInterval = setInterval(updateText, 500);

    while (true) {
        const ready = await checkHealth();
        if (ready) {
            clearInterval(textInterval);
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
            inputEmpty.focus();
            return;
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
}

// Initialize
async function init() {
    // Wait for model to be ready first
    await waitForModel();

    formEmpty.addEventListener('submit', handleSubmit);
    formChat.addEventListener('submit', handleSubmit);
    inputEmpty.addEventListener('keydown', handleKeydown);
    inputChat.addEventListener('keydown', handleKeydown);
    inputEmpty.addEventListener('input', handleInput);
    inputChat.addEventListener('input', handleInput);
    messagesEl.addEventListener('click', handleCopy);

    // Initialize textarea heights
    autoResizeTextarea(inputEmpty);
    autoResizeTextarea(inputChat);
}

// Get active input
function getActiveInput() {
    return messages.length === 0 ? inputEmpty : inputChat;
}

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();

    const input = getActiveInput();
    const content = input.value.trim();
    if (!content || isGenerating) return;

    // Add user message
    messages.push({ role: 'user', content });
    input.value = '';
    autoResizeTextarea(input);

    // Switch to chat state
    if (messages.length === 1) {
        emptyState.classList.add('hidden');
        chatState.classList.remove('hidden');
        inputChat.focus();
    }

    renderMessages();

    // Add placeholder for assistant response
    const assistantIndex = messages.length;
    messages.push({ role: 'assistant', content: '', thinking: '', isStreaming: true, isThinking: true });
    renderMessages();

    // Disable input while generating
    setGenerating(true);

    try {
        await streamResponse(assistantIndex);
    } catch (error) {
        console.error('Generation error:', error);
        messages[assistantIndex].content = `Error: ${error.message}`;
        messages[assistantIndex].isError = true;
    } finally {
        messages[assistantIndex].isStreaming = false;
        setGenerating(false);
        renderMessages();
    }
}

// Stream response from server
async function streamResponse(assistantIndex) {
    const startTime = performance.now();

    const requestBody = {
        messages: messages.slice(0, -1).map(m => ({
            role: m.role,
            content: m.content
        })),
        stream: true,
        max_completion_tokens: 2048,
        temperature: 0.7
    };

    const response = await fetch(`${API_BASE}/api/v1/chat/completions/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error?.message || 'Request failed');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let tokenCount = 0;
    let firstToken = true;
    let ttft = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;

            const jsonStr = line.slice(6);
            if (jsonStr === '[DONE]') {
                reader.cancel();
                return;
            }

            try {
                const data = JSON.parse(jsonStr);
                const content = data.choices?.[0]?.delta?.content;
                const thinking = data.choices?.[0]?.delta?.thinking;
                const thinkingDone = data.choices?.[0]?.delta?.thinking_done;

                // Stream thinking tokens
                if (thinking) {
                    messages[assistantIndex].thinking += thinking;
                    renderMessages();
                }

                // Thinking done - collapse the section
                if (thinkingDone) {
                    messages[assistantIndex].isThinking = false;
                    renderMessages();
                }

                if (content) {
                    if (firstToken) {
                        ttft = performance.now() - startTime;
                        firstToken = false;
                    }

                    messages[assistantIndex].content += content;
                    tokenCount++;
                    renderMessages();
                }

                if (data.choices?.[0]?.finish_reason) {
                    break;
                }
            } catch (e) {
                console.warn('Failed to parse SSE data:', e);
            }
        }
    }

    const totalTime = performance.now() - startTime;
    const tokensPerSecond = tokenCount / (totalTime / 1000);
    statsEl.textContent = `${tokenCount} tokens | ${tokensPerSecond.toFixed(1)} tok/s | TTFT: ${ttft.toFixed(0)}ms`;
}

// Handle input changes
function handleInput(e) {
    autoResizeTextarea(e.target);
}

// Auto-resize textarea
function autoResizeTextarea(textarea) {
    // Reset height to min-height to get the correct scrollHeight
    textarea.style.height = '72px';

    // Set the height to match content, with a minimum of 72px (3 lines)
    // and maximum of 160px (then scroll kicks in)
    const minHeight = 48;
    const maxHeight = 160;
    const scrollHeight = textarea.scrollHeight;

    if (scrollHeight > minHeight) {
        textarea.style.height = Math.min(scrollHeight, maxHeight) + 'px';
    }
}

// Handle keyboard shortcuts
function handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const form = e.target.closest('form');
        form.dispatchEvent(new Event('submit'));
    }
}

// Handle copy button clicks
async function handleCopy(e) {
    const btn = e.target.closest('.copy-btn');
    if (!btn) return;

    const index = parseInt(btn.dataset.index);
    const content = messages[index]?.content;
    if (!content) return;

    try {
        await navigator.clipboard.writeText(content);
        btn.classList.add('copied');
        setTimeout(() => btn.classList.remove('copied'), 1500);
    } catch (err) {
        console.error('Copy failed:', err);
    }
}

// Set generating state
function setGenerating(generating) {
    isGenerating = generating;
    const btns = document.querySelectorAll('.send-btn');
    const inputs = document.querySelectorAll('textarea');
    btns.forEach(btn => btn.disabled = generating);
    inputs.forEach(input => input.disabled = generating);
}

// Render messages
function renderMessages() {
    messagesEl.innerHTML = messages.map((m, i) => {
        const content = m.role === 'assistant'
            ? marked.parse(m.content || '')
            : escapeHtml(m.content);

        let html = `
            <div class="message ${m.role}${m.isError ? ' error' : ''}">`;

        // Add thinking section for assistant messages with thinking content
        if (m.role === 'assistant' && m.thinking) {
            const isOpen = m.isThinking ? ' open' : '';
            const label = m.isThinking ? 'Thinking...' : 'Thought for a brief moment';
            html += `
                <details class="thinking-section"${isOpen}>
                    <summary>
                        <svg class="thinking-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <path d="M12 16v-4"></path>
                            <path d="M12 8h.01"></path>
                        </svg>
                        <span>${label}</span>
                    </summary>
                    <div class="thinking-content">${marked.parse(m.thinking)}</div>
                </details>`;
        }

        html += `<div class="content">${content}`;

        if (m.isStreaming && !m.content) {
            html += `
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>`;
        }

        html += `</div>`;

        if (m.role === 'assistant' && m.content && !m.isStreaming) {
            html += `
                <div class="message-actions">
                    <button class="copy-btn" data-index="${i}" title="Copy">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                    </button>
                </div>`;
        }

        html += `</div>`;
        return html;
    }).join('');

    // Add copy buttons to code blocks
    messagesEl.querySelectorAll('pre').forEach(pre => {
        if (pre.querySelector('.code-copy-btn')) return;
        const btn = document.createElement('button');
        btn.className = 'code-copy-btn';
        btn.title = 'Copy code';
        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>`;
        btn.onclick = async () => {
            const code = pre.querySelector('code')?.textContent || pre.textContent;
            try {
                await navigator.clipboard.writeText(code);
                btn.classList.add('copied');
                setTimeout(() => btn.classList.remove('copied'), 1500);
            } catch (err) {
                console.error('Copy failed:', err);
            }
        };
        pre.appendChild(btn);
    });

    requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });
}

// Escape HTML
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Start
init();
