# Imports here
import os
import re 
import random
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional

from flask import Flask, render_template, request, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.middleware.proxy_fix import ProxyFix

import chatbot

# Config logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# App Configuration Settings
class Config:
    """Application configuration with secure defaults"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')

    # path = "user_and_history/User"
    # CHAT_ROOMS = []

    # # dirs=directories
    # for (root, dirs, file) in os.walk(path):
    #     for f in file:
    #         if '.txt' in f:
    #             CHAT_ROOMS.append(f.replace('.txt', ''))
    
    # Available chat rooms - stored as constant for now, could be moved to database
    # CHAT_ROOMS = [
    #     'General',
    #     'Zero to Knowing',
    #     'Code with Josh',
    #     'The Nerd Nook'
    # ]

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Handle reverse proxy headers
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Initialize SocketIO with appropriate CORS settings
socketio = SocketIO(
    app,
    cors_allowed_origins=app.config['CORS_ORIGINS'],
    logger=True,
    engineio_logger=True
)

# In-memory storage for active users
# In production, consider using Redis or another distributed storage
active_users: Dict[str, dict] = {}

def get_rooms():

    rooms = []

    for (root, dirs, file) in os.walk('user_and_history/User'):
            for f in file:
                if '.txt' in f:
                    rooms.append(f.replace('.txt', ''))

    return rooms

@app.route('/')
def index():
    if 'username' not in session:
        session['username'] = 'User'
        logger.info(f"New user session created: {session['username']}")
    
    return render_template(
        'index.html',
        username=session['username'],
        # rooms=app.config['CHAT_ROOMS']
        rooms = get_rooms()
    )

@socketio.event
def connect():
    try:
        if 'username' not in session:
            session['username'] = 'User'
        
        active_users[request.sid] = {
            'username': session['username'],
            'connected_at': datetime.now().isoformat()
        }
        
        # emit('active_users', {
        #     'users': [user['username'] for user in active_users.values()]
        # }, broadcast=True)
        
        logger.info(f"User connected: {session['username']}")
    
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        return False

@socketio.event
def disconnect():
    try:
        if request.sid in active_users:
            username = active_users[request.sid]['username']
            del active_users[request.sid]
            
            emit('active_users', {
                'users': [user['username'] for user in active_users.values()]
            }, broadcast=True)
            
            logger.info(f"User disconnected: {username}")
    
    except Exception as e:
        logger.error(f"Disconnection error: {str(e)}")

@socketio.on('join')
def on_join(data: dict):
    try:
        username = session['username']
        room = data['room']

        if room not in get_rooms():
        # if room not in app.config['CHAT_ROOMS']:
            logger.warning(f"Invalid room join attempt: {room}")
            return
        
        join_room(room)
        active_users[request.sid]['room'] = room
        
        with open(f"user_and_history/User/{room}.txt", "r", encoding="utf-8") as f:
            content = f.read()
        chat = re.split(r'(?i)(?=\b(?:user|bot):)', content, flags=re.MULTILINE)
        chat = [item.strip() for item in chat if item.strip()]
        for text in chat:
            if text.startswith(f'{username}:'):
                emit('message', {
                    'msg': text.replace(f'{username}:', ''),
                    'username': username,
                    'room': room,
                    'timestamp': datetime.now().isoformat()
                }, room=room)
            elif text.startswith('Bot:'):
                emit('message', {
                    'msg': text.replace('Bot:', ''),
                    'username': 'Bot',
                    'room': room,
                    'timestamp': datetime.now().isoformat()
                }, room=room)

        # emit('status', {
        #     'msg': f'{username} has joined the room.',
        #     'type': 'join',
        #     'timestamp': datetime.now().isoformat()
        # }, room=room)
        
        logger.info(f"User {username} joined room: {room}")
    
    except Exception as e:
        logger.error(f"Join room error: {str(e)}")

@socketio.on('leave')
def on_leave(data: dict):
    try:
        username = session['username']
        room = data['room']
        
        leave_room(room)
        if request.sid in active_users:
            active_users[request.sid].pop('room', None)
        
        emit('status', {
            'msg': f'{username} has left the room.',
            'type': 'leave',
            'timestamp': datetime.now().isoformat()
        }, room=room)
        
        logger.info(f"User {username} left room: {room}")
    
    except Exception as e:
        logger.error(f"Leave room error: {str(e)}")

@socketio.on('message')
def handle_message(data: dict):
    try:
        username = session['username']
        room = data.get('room', 'General')
        msg_type = data.get('type', 'message')
        message = data.get('msg', '').strip()
        
        if not message:
            return
        
        timestamp = datetime.now().isoformat()

        if room != 'General':
            emit('message', {
                    'msg': message,
                    'username': username,
                    'room': room,
                    'timestamp': timestamp
                }, room=room)
                    
            bot_answer = chatbot.chat_with_bot(message, room)
            emit('message', {
                    'msg': bot_answer,
                    'username': 'Bot',
                    'room': room,
                    'timestamp': timestamp
                }, room=room)
            
            with open(f"user_and_history/User/{room}.txt", "a", encoding="utf-8") as f:
                f.write(f'User:{message}\n')

            with open(f"user_and_history/User/{room}.txt", "a", encoding="utf-8") as f:
                f.write(f'Bot:{bot_answer}\n')    
            
            logger.info(f"Message sent in {room} by {username}")
    
    except Exception as e:
        logger.error(f"Message handling error: {str(e)}")

@socketio.on('create_new_chat')
def crete_chat(data):
    room = data['roomName']
    json_data = {"last_diseases" : ""}
    json_str = json.dumps(json_data, indent=4)
    with open(f"user_and_history/User/{room}.json", "w", encoding="utf-8") as f:
        f.write(json_str)
    with open(f"user_and_history/User/{room}.txt", "x", encoding="utf-8"):
        pass
    # get_rooms()
    # return redirect(url_for("index"))
    # app.config["CHAT_ROOMS"].append(room)

    emit(
        "new_chat",
        {
            "room": room
        },
        broadcast=True
    )

@socketio.on('edit-name')
def editRoomName(data):
    room = data['room']
    newName = data['newName']
    os.rename(f"user_and_history/User/{room}.txt",f"user_and_history/User/{newName}.txt")
    os.rename(f"user_and_history/User/{room}.json",f"user_and_history/User/{newName}.json")

@socketio.on('delete')
def deleteRoom(data):
    room = data['room']
    os.remove(f"user_and_history/User/{room}.txt")
    os.remove(f"user_and_history/User/{room}.json")

if __name__ == '__main__':
    # In production, use gunicorn or uwsgi instead
    port = int(os.environ.get('PORT', 5000))
    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=app.config['DEBUG'],
        use_reloader=app.config['DEBUG']
    )