// Nexon Chat Application
const API_BASE = window.location.origin;

// Greeting prompts
const GREETINGS = [
    "What's on your mind?",
    "What are you curious about?",
    "What would you like to explore?",
    "What are you thinking about?"
];

function getRandomGreeting() {
    return GREETINGS[Math.floor(Math.random() * GREETINGS.length)];
}

// State
let messages = [];
let isGenerating = false;
let currentAbortController = null;
let currentModel = '';

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
const modelNameEl = document.getElementById('model-name');

// Configure marked with syntax highlighting
marked.use({
    breaks: true,
    gfm: true,
    renderer: {
        code(code, lang) {
            const text = typeof code === 'object' ? code.text : code;
            const language = typeof code === 'object' ? code.lang : lang;

            let highlighted;
            if (language && hljs.getLanguage(language)) {
                highlighted = hljs.highlight(text, { language }).value;
            } else {
                highlighted = hljs.highlightAuto(text).value;
            }
            return `<pre><code class="hljs language-${language || ''}">${highlighted}</code></pre>`;
        }
    }
});

// Check if model is ready
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        if (data.model) {
            currentModel = data.model;
        }
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

            // Set random greeting
            const greetingEl = document.querySelector('.greeting-text');
            if (greetingEl) greetingEl.textContent = getRandomGreeting();

            inputEmpty.focus();

            // Display model name
            if (modelNameEl && currentModel) {
                const shortName = currentModel.split('/').pop();
                modelNameEl.textContent = shortName;
                modelNameEl.title = currentModel;
            }
            return;
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
}

// Initialize theme
function initTheme() {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.documentElement.setAttribute('data-theme', 'dark');
        updateThemeIcon(true);
    }
}

// Toggle theme
function toggleTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    if (isDark) {
        document.documentElement.removeAttribute('data-theme');
        localStorage.setItem('theme', 'light');
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
    }
    updateThemeIcon(!isDark);
}

// Update theme icon
function updateThemeIcon(isDark) {
    document.querySelectorAll('.sun-icon').forEach(el => el.classList.toggle('hidden', isDark));
    document.querySelectorAll('.moon-icon').forEach(el => el.classList.toggle('hidden', !isDark));
}

// Initialize
async function init() {
    // Init theme first (before waiting for model)
    initTheme();

    // Wait for model to be ready
    await waitForModel();

    // Form submissions
    formEmpty.addEventListener('submit', handleSubmit);
    formChat.addEventListener('submit', handleSubmit);

    // Input events
    inputEmpty.addEventListener('keydown', handleKeydown);
    inputChat.addEventListener('keydown', handleKeydown);
    inputEmpty.addEventListener('input', handleInput);
    inputChat.addEventListener('input', handleInput);

    // Message actions
    messagesEl.addEventListener('click', handleMessageClick);

    // Header buttons
    document.getElementById('new-chat-btn')?.addEventListener('click', resetChat);
    document.getElementById('theme-btn')?.addEventListener('click', toggleTheme);
    document.getElementById('export-btn')?.addEventListener('click', exportChat);

    // Stop buttons
    document.querySelectorAll('.stop-btn').forEach(btn => {
        btn.addEventListener('click', handleStop);
    });

    // Global keyboard shortcuts
    document.addEventListener('keydown', handleGlobalKeydown);

    // Initialize textarea heights
    autoResizeTextarea(inputEmpty);
    autoResizeTextarea(inputChat);

    // Initialize char counts
    updateCharCount(inputEmpty);
    updateCharCount(inputChat);
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
    updateCharCount(input);

    // Switch to chat state
    if (messages.length === 1) {
        emptyState.classList.add('hidden');
        chatState.classList.remove('hidden');
        inputChat.focus();
    }

    await generateResponse();
}

// Generate response
async function generateResponse() {
    renderMessages();

    // Add placeholder for assistant response
    const assistantIndex = messages.length;
    messages.push({ role: 'assistant', content: '', thinking: '', isStreaming: true, isThinking: true, showThinking: true });
    renderMessages();

    // Disable input while generating
    setGenerating(true);

    try {
        await streamResponse(assistantIndex);
    } catch (error) {
        if (error.name === 'AbortError') {
            messages[assistantIndex].content += '\n\n*[Generation stopped]*';
        } else {
            console.error('Generation error:', error);
            messages[assistantIndex].content = `Error: ${error.message}`;
            messages[assistantIndex].isError = true;
        }
    } finally {
        messages[assistantIndex].isStreaming = false;
        setGenerating(false);
        renderMessages();
    }
}

// Stream response from server
async function streamResponse(assistantIndex) {
    const startTime = performance.now();

    // Create abort controller
    currentAbortController = new AbortController();

    const requestBody = {
        messages: messages.slice(0, -1).map(m => ({
            role: m.role,
            content: m.content
        })),
        stream: true,
        temperature: 0.7
    };

    const response = await fetch(`${API_BASE}/api/v1/chat/completions/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
        signal: currentAbortController.signal
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

    try {
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
                        updateStreamingMessage(assistantIndex);
                    }

                    // Thinking done - just update state
                    if (thinkingDone) {
                        messages[assistantIndex].isThinking = false;
                        updateStreamingMessage(assistantIndex);
                    }

                    if (content) {
                        if (firstToken) {
                            ttft = performance.now() - startTime;
                            firstToken = false;
                        }

                        messages[assistantIndex].content += content;
                        tokenCount++;
                        updateStreamingMessage(assistantIndex);
                    }

                    if (data.choices?.[0]?.finish_reason) {
                        break;
                    }
                } catch (e) {
                    console.warn('Failed to parse SSE data:', e);
                }
            }
        }
    } finally {
        reader.releaseLock();
        currentAbortController = null;
    }

    const totalTime = performance.now() - startTime;
    const tokensPerSecond = tokenCount / (totalTime / 1000);
    statsEl.textContent = `${tokenCount} tokens | ${tokensPerSecond.toFixed(1)} tok/s | TTFT: ${ttft.toFixed(0)}ms`;
}

// Handle stop button
function handleStop() {
    if (currentAbortController) {
        currentAbortController.abort();
    }
}

// Reset chat (new chat)
function resetChat() {
    // Stop any ongoing generation
    handleStop();

    messages = [];
    statsEl.textContent = '';

    chatState.classList.add('hidden');
    emptyState.classList.remove('hidden');

    // Set new random greeting
    const greetingEl = document.querySelector('.greeting-text');
    if (greetingEl) greetingEl.textContent = getRandomGreeting();

    inputEmpty.value = '';
    inputChat.value = '';
    autoResizeTextarea(inputEmpty);
    autoResizeTextarea(inputChat);
    updateCharCount(inputEmpty);
    updateCharCount(inputChat);

    inputEmpty.focus();
}

// Export chat as markdown
function exportChat() {
    if (messages.length === 0) return;

    let markdown = '# Chat Export\n\n';
    markdown += `*Exported from Nexon on ${new Date().toLocaleString()}*\n\n---\n\n`;

    for (const msg of messages) {
        if (msg.role === 'user') {
            markdown += `**User:**\n${msg.content}\n\n`;
        } else {
            markdown += `**Assistant:**\n${msg.content}\n\n`;
        }
    }

    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
}

// Handle input changes
function handleInput(e) {
    autoResizeTextarea(e.target);
    updateCharCount(e.target);
}

// Update character count
function updateCharCount(textarea) {
    const form = textarea.closest('form');
    const countEl = form?.querySelector('.char-count');
    if (countEl) {
        const len = textarea.value.length;
        countEl.textContent = len > 0 ? `${len}` : '';
    }
}

// Auto-resize textarea
function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    const minHeight = 24;
    const maxHeight = 160;
    const scrollHeight = textarea.scrollHeight;
    textarea.style.height = Math.max(minHeight, Math.min(scrollHeight, maxHeight)) + 'px';
}

// Handle keyboard shortcuts in textarea
function handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const form = e.target.closest('form');
        form.dispatchEvent(new Event('submit'));
    }
}

// Handle global keyboard shortcuts
function handleGlobalKeydown(e) {
    // Escape - stop generation
    if (e.key === 'Escape' && isGenerating) {
        handleStop();
    }

    // Cmd/Ctrl + Shift + O - new chat
    if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'o') {
        e.preventDefault();
        resetChat();
    }
}

// Handle message click events (copy)
function handleMessageClick(e) {
    // Copy button
    const copyBtn = e.target.closest('.copy-btn');
    if (copyBtn) {
        const index = parseInt(copyBtn.dataset.index);
        const content = messages[index]?.content;
        if (content) {
            navigator.clipboard.writeText(content).then(() => {
                copyBtn.classList.add('copied');
                setTimeout(() => copyBtn.classList.remove('copied'), 1500);
            });
        }
        return;
    }
}

// Set generating state
function setGenerating(generating) {
    isGenerating = generating;

    // Toggle send/stop buttons
    document.querySelectorAll('.send-btn').forEach(btn => {
        btn.classList.toggle('hidden', generating);
    });
    document.querySelectorAll('.stop-btn').forEach(btn => {
        btn.classList.toggle('hidden', !generating);
    });

    // Disable/enable inputs
    document.querySelectorAll('textarea').forEach(input => {
        input.disabled = generating;
    });
}

// Update only the streaming message content (optimized for smooth animations)
function updateStreamingMessage(index) {
    const messageEl = messagesEl.children[index];
    if (!messageEl) {
        renderMessages();
        return;
    }

    const m = messages[index];

    // Update thinking content if present
    const thinkingSection = messageEl.querySelector('.thinking-section');
    const thinkingContent = messageEl.querySelector('.thinking-content');

    // If we need a thinking section but don't have one, do full render
    if ((m.thinking || m.isThinking) && !thinkingSection) {
        renderMessages();
        return;
    }

    if (thinkingContent && m.thinking) {
        thinkingContent.innerHTML = marked.parse(m.thinking);
    }

    // Update thinking section class (active/inactive)
    if (thinkingSection) {
        if (m.isThinking) {
            thinkingSection.classList.add('active');
            const label = thinkingSection.querySelector('summary > span');
            if (label) label.textContent = 'Thinking...';
        } else {
            thinkingSection.classList.remove('active');
            const label = thinkingSection.querySelector('summary > span');
            if (label) label.textContent = 'Thought';
        }
    }

    // Update main content
    const contentEl = messageEl.querySelector('.content');
    if (contentEl) {
        const typingIndicator = contentEl.querySelector('.typing-indicator');

        if (m.content) {
            contentEl.innerHTML = marked.parse(m.content);
            // Add copy buttons to new code blocks
            contentEl.querySelectorAll('pre').forEach(pre => {
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
        } else if (m.isStreaming && !typingIndicator) {
            // Show typing indicator if no content yet
            contentEl.innerHTML = `
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>`;
        }
    }

    // Auto-scroll
    requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });
}

// Render messages
function renderMessages() {
    // Preserve open state of thinking sections
    const openThinking = new Set();
    messagesEl.querySelectorAll('.thinking-section').forEach((el, i) => {
        if (el.open) openThinking.add(i);
    });

    messagesEl.innerHTML = messages.map((m, i) => {
        const content = m.role === 'assistant'
            ? marked.parse(m.content || '')
            : escapeHtml(m.content);

        let html = `<div class="message ${m.role}${m.isError ? ' error' : ''}">`;

        // Add thinking indicator (collapsed, click to expand)
        if (m.role === 'assistant' && (m.thinking || m.isThinking) && m.showThinking !== false) {
            const thinkingClass = m.isThinking ? 'thinking-section active' : 'thinking-section';
            const thinkingLabel = m.isThinking ? 'Thinking...' : 'Thought';
            html += `
                <details class="${thinkingClass}">
                    <summary>
                        <svg class="thinking-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <path d="M12 16v-4"></path>
                            <path d="M12 8h.01"></path>
                        </svg>
                        <span>${thinkingLabel}</span>
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

        // Assistant message actions (copy, retry)
        if (m.role === 'assistant' && !m.isStreaming && m.content) {
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

    // Restore open state of thinking sections
    messagesEl.querySelectorAll('.thinking-section').forEach((el, i) => {
        if (openThinking.has(i)) el.open = true;
    });

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

// Speech Recognition
let recognition = null;
let isRecording = false;

function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        // Hide mic buttons if speech recognition not supported
        document.querySelectorAll('.mic-btn').forEach(btn => btn.style.display = 'none');
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }

        // Get the active input field
        const activeInput = document.activeElement.tagName === 'TEXTAREA'
            ? document.activeElement
            : (emptyState.classList.contains('hidden') ? inputChat : inputEmpty);

        // Update with transcript
        if (activeInput) {
            activeInput.value = transcript;
            activeInput.dispatchEvent(new Event('input'));
        }
    };

    recognition.onend = () => {
        if (isRecording) {
            // Restart if still recording (browser may stop after silence)
            recognition.start();
        }
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        stopRecording();
    };
}

function startRecording(btn) {
    if (!recognition) return;

    isRecording = true;
    btn.classList.add('recording');

    try {
        recognition.start();
    } catch (e) {
        // Already started
    }
}

function stopRecording() {
    if (!recognition) return;

    isRecording = false;
    document.querySelectorAll('.mic-btn').forEach(btn => btn.classList.remove('recording'));

    try {
        recognition.stop();
    } catch (e) {
        // Already stopped
    }
}

// Set up mic button handlers
document.querySelectorAll('.mic-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording(btn);
        }
    });
});

// Initialize speech recognition
initSpeechRecognition();

// Start
init();
