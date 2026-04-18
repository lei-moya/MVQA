// src/main.js
import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from './router'
import '@fortawesome/fontawesome-free/css/all.css'

// 添加全局错误处理，忽略WebSocket相关错误
window.addEventListener('error', function(event) {
  // 忽略socket.send()相关的错误
  if (event.message && event.message.includes('socket.send() raised exception')) {
    event.preventDefault();
    return false;
  }
  // 其他错误正常处理
  return true;
});

// 忽略WebSocket连接错误
if (window.WebSocket) {
  const originalWebSocket = window.WebSocket;
  window.WebSocket = function(...args) {
    const ws = new originalWebSocket(...args);
    // 忽略错误事件
    ws.addEventListener('error', function(event) {
      // 可以在这里添加日志或其他处理
    });
    return ws;
  };
  window.WebSocket.prototype = originalWebSocket.prototype;
  window.WebSocket.CONNECTING = originalWebSocket.CONNECTING;
  window.WebSocket.OPEN = originalWebSocket.OPEN;
  window.WebSocket.CLOSING = originalWebSocket.CLOSING;
  window.WebSocket.CLOSED = originalWebSocket.CLOSED;
}

const app = createApp(App)
app.use(router)
app.use(ElementPlus)
app.mount('#app')
