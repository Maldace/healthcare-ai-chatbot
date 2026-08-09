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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Application configuration with secure defaults"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')

app = Flask(__name__)
app.config.from_object(Config)

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

socketio = SocketIO(
    app,
    cors_allowed_origins=app.config['CORS_ORIGINS'],
    logger=True,
    engineio_logger=True
)

active_users: Dict[str, dict] = {}

def get_rooms(user):

    rooms = []

    for (root, dirs, file) in os.walk(f'user_and_history/{user}'):
            for f in file:
                if '.txt' in f:
                    rooms.append(f.replace('.txt', ''))

    return rooms

# @app.route('/')
# def loginRedirect():
    
#     return render_template(
#         'login.html'
#     )
    
# @socketio.on('login')
# def login(data):
#     username = data['username']
#     password = data['password']
#     session['username'] = username
#     return {'success': True}

@app.route('/')
def index():
    if 'username' not in session:
        # return redirect('/login')
        session['username'] = 'User'
        logger.info(f"New user session created: {session['username']}")
    
    return render_template(
        'index.html',
        username=session['username'],
        # rooms=app.config['CHAT_ROOMS']
        rooms = get_rooms(session['username'])
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

        if room not in get_rooms(username):
            logger.warning(f"Invalid room join attempt: {room}")
            return
        
        join_room(room)
        active_users[request.sid]['room'] = room
        
        with open(f"user_and_history/{username}/{room}.txt", "r", encoding="utf-8") as f:
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
                    
            bot_answer = chatbot.chat_with_bot(message, room, username)
            emit('message', {
                    'msg': bot_answer,
                    'username': 'Bot',
                    'room': room,
                    'timestamp': timestamp
                }, room=room)
            
            with open(f"user_and_history/{username}/{room}.txt", "a", encoding="utf-8") as f:
                f.write(f'User:{message}\n')

            with open(f"user_and_history/{username}/{room}.txt", "a", encoding="utf-8") as f:
                f.write(f'Bot:{bot_answer}\n')    
            
            logger.info(f"Message sent in {room} by {username}")
    
    except Exception as e:
        logger.error(f"Message handling error: {str(e)}")

@socketio.on('create_new_chat')
def crete_chat(data):
    room = data['roomName']
    username = session['username']
    json_data = {"last_diseases" : ""}
    json_str = json.dumps(json_data, indent=4)
    with open(f"user_and_history/{username}/{room}.json", "w", encoding="utf-8") as f:
        f.write(json_str)
    with open(f"user_and_history/{username}/{room}.txt", "x", encoding="utf-8"):
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
    username = session['username']
    os.rename(f"user_and_history/{username}/{room}.txt",f"user_and_history/{username}/{newName}.txt")
    os.rename(f"user_and_history/{username}/{room}.json",f"user_and_history/{username}/{newName}.json")

@socketio.on('delete')
def deleteRoom(data):
    room = data['room']
    username = session['username']
    os.remove(f"user_and_history/{username}/{room}.txt")
    os.remove(f"user_and_history/{username}/{room}.json")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=app.config['DEBUG'],
        use_reloader=app.config['DEBUG']
    )