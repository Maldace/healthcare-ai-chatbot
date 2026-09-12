# Imports here
import os
import re 
import logging
import json
from datetime import datetime, timedelta
from typing import Dict

from flask import Flask, render_template, request, session, redirect, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.middleware.proxy_fix import ProxyFix

import pyodbc

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

DRIVER_NAME = 'ODBC Driver 17 for SQL Server'
SERVER_NAME = r'LAPTOP-AEBT61JV\SQLEXPRESS'
DATABASE_NAME = 'chatbot_db'
connection_string = f'DRIVER={DRIVER_NAME};SERVER={SERVER_NAME};DATABASE={DATABASE_NAME};Trusted_Connection=yes;'
conn = pyodbc.connect(connection_string)


app = Flask(__name__)
app.config.from_object(Config)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.secret_key='mysecretkey'

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

    # for (root, dirs, file) in os.walk(f'../chatbot_ai_user_and_history/user_and_history/{user}'):
    #         for f in file:
    for path, subdirs, files in os.walk(f'../chatbot_ai_user_and_history/user_and_history/{user}'):
        for name in sorted(files, key=lambda name: os.path.getmtime(os.path.join(path, name)), reverse=True):
            if '.txt' in name:
                rooms.append(name.replace('.txt', ''))

    return rooms
    

@app.route('/')
def index():
    try:
        if 'username' not in session:
            return render_template('login.html')
        
        return render_template(
            'index.html',
            username = session['username'],
            rooms = get_rooms(session['username'])
        )
    except Exception as e:
        logger.error(f"Redirect index error: {str(e)}")
        return render_template('login.html', error='Có lỗi, vui lòng đăng nhập lại!')

# @app.route('/login')
# def loginRedirect():
#     return render_template(
#         'login.html'
#     )
    
@app.route('/login', methods = ['GET', 'POST'])
def login():
    try:
        if request.method == 'POST':
            typedUsername = request.form['username']
            cursor = conn.cursor()
            sql = "select password from tblUser where username like ?"
            cursor.execute(sql, (typedUsername))
            realPassword = cursor.fetchone()
            if realPassword==None:
                flash('Tài khoản hoặc mật khẩu không đúng. Vui lòng thử lại!')
            else:
                typedPassword = request.form['password']
                if typedPassword==realPassword[0]:
                    session['username'] = request.form['username']
                    session.permanent = True
                    return redirect('/')
                else:
                    flash('Tài khoản hoặc mật khẩu không đúng. Vui lòng thử lại!')
            return render_template('login.html')
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return render_template('login.html', error='Có lỗi, vui lòng đăng nhập lại!')

@app.route('/logout')
def logout():
    try:
        session.pop('username', None)
        return redirect('/')
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return render_template('login.html', error='Có lỗi, vui lòng đăng nhập lại!')

@app.route('/register_redirect')
def register_redirect():
    try:
        return render_template('register.html')
    except Exception as e:
        logger.error(f"Redirect register error: {str(e)}")
        return render_template('login.html', error='Có lỗi, vui lòng đăng nhập lại!')

@app.route('/register', methods = ['GET', 'POST'])
def register():
    try:
        if request.method == 'POST':
            if request.form['password'] == request.form['confirmPassword']:
                username = request.form['username']
                password = request.form['password']
                email = request.form['email']
                phone = request.form['phone']
                sql = "select username from tblUser where username like ?"
                cursor = conn.cursor()
                cursor.execute(sql, (username))
                existingUser = cursor.fetchone()
                if existingUser == None:
                    sql = 'insert into tblUser (username, password, email, phone) values (?, ?, ?, ?)'
                    cursor.execute(sql, (username, password, email, phone))
                    os.mkdir(f'../chatbot_ai_user_and_history/user_and_history/{username}')
                    with open(f"../chatbot_ai_user_and_history/user_and_history/{username}/General.txt", "x", encoding="utf-8"):
                        pass
                    general="""User:Welcome to the Healthcare Chatbot AI!
Bot:Here you can ask the chatbot about symptoms, medicine, diet,... of some diseases.
User:Be careful, as the information given by chatbot can be miscorrect and is just for reference only, cannot replace doctor's instruction.
Bot:So, enjoy!!
                    """
                    with open(f"../chatbot_ai_user_and_history/user_and_history/{username}/General.txt", "a", encoding="utf-8") as f:
                        f.write(general)
                    flash('Đăng ký tài khoản thành công. Vui lòng đăng nhập để sử dụng dịch vụ')
                    return redirect('/')
                else:
                    flash('Tên tài khoản đã được sử dụng. Vui lòng chọn tên khác')
            else:
                flash('Mật khẩu nhập lại không đúng. Vui lòng thử lại!')
            return render_template('register.html')
    except Exception as e:
        logger.error(f"Register error: {str(e)}")
        return render_template('login.html', error='Có lỗi, vui lòng đăng nhập lại!')

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
        
        with open(f"../chatbot_ai_user_and_history/user_and_history/{username}/{room}.txt", "r", encoding="utf-8") as f:
            content = f.read()
        chat = re.split(r'(?i)(?=\b(?:user|bot):)', content, flags=re.MULTILINE)
        chat = [item.strip() for item in chat if item.strip()]
        for text in chat:
            # if text.startswith(f'{username}:'):
            #     emit('message', {
            #         'msg': text.replace(f'{username}:', ''),
            #         'username': username,
            #         'room': room,
            #         'timestamp': datetime.now().isoformat()
            #     }, room=room)
            # elif text.startswith('Bot:'):
            #     emit('message', {
            #         'msg': text.replace('Bot:', ''),
            #         'username': 'Bot',
            #         'room': room,
            #         'timestamp': datetime.now().isoformat()
            #     }, room=room)
            sender, msg = text.split(':', 1)
            # print(sender)
            # emit('message', {
            #     'msg': msg,
            #     'username': sender,
            #     'room': room,
            #     'timestamp': datetime.now().isoformat()
            #     }, room=room)
            if sender==('Bot'):
                emit('message', {
                    'msg': msg,
                    'username': 'Bot',
                    'room': room,
                    'timestamp': datetime.now().isoformat()
                }, room=room)
            else:
                emit('message', {
                    'msg': msg,
                    'username': username,
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
            
            with open(f"../chatbot_ai_user_and_history/user_and_history/{username}/{room}.txt", "a", encoding="utf-8") as f:
                f.write(f'User:{message}\n')

            with open(f"../chatbot_ai_user_and_history/user_and_history/{username}/{room}.txt", "a", encoding="utf-8") as f:
                f.write(f'Bot:{bot_answer}\n')    
            
            logger.info(f"Message sent in {room} by {username}")
    
    except Exception as e:
        logger.error(f"Message handling error: {str(e)}")

@socketio.on('create_new_chat')
def create_chat(data):
    try:
        room = data['roomName']
        username = session['username']
        json_data = {"last_diseases" : ""}
        json_str = json.dumps(json_data, indent=4)
        with open(f"../chatbot_ai_user_and_history/user_and_history/{username}/{room}.json", "w", encoding="utf-8") as f:
            f.write(json_str)
        with open(f"../chatbot_ai_user_and_history/user_and_history/{username}/{room}.txt", "x", encoding="utf-8"):
            pass

        emit(
            "new_chat",
            {
                "room": room
            },
            broadcast=True
        )
    except Exception as e:
        logger.error(f"Create chat error: {str(e)}")

@socketio.on('edit-name')
def edit_room_name(data):
    try:
        room = data['room']
        newName = data['newName']
        username = session['username']
        os.rename(f"../chatbot_ai_user_and_history/user_and_history/{username}/{room}.txt",f"../chatbot_ai_user_and_history/user_and_history/{username}/{newName}.txt")
        os.rename(f"../chatbot_ai_user_and_history/user_and_history/{username}/{room}.json",f"../chatbot_ai_user_and_history/user_and_history/{username}/{newName}.json")
    except Exception as e:
        logger.error(f"Edit room name error: {str(e)}")

@socketio.on('delete')
def delete_room(data):
    try:
        room = data['room']
        username = session['username']
        os.remove(f"../chatbot_ai_user_and_history/user_and_history/{username}/{room}.txt")
        os.remove(f"../chatbot_ai_user_and_history/user_and_history/{username}/{room}.json")
    except Exception as e:
        logger.error(f"Delete room error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=app.config['DEBUG'],
        use_reloader=app.config['DEBUG']
    )