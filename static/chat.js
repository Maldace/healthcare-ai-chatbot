let socket = io();
let currentRoom = 'General';
let username = document.getElementById('username').textContent;
let roomMessages = {};

socket.on('connect', () => {
	joinRoom('General');
	highlightActiveRoom('General');
});

socket.on('message', (data) => {
	addMessage(
		data.username,
		data.msg,
		data.sender === username ? 'own' : 'other',
		data.sender,
		data.room
	);
});

// socket.on('status', (data) => {
// 	addMessage('System', data.msg, 'system');
// });

// socket.on('active_users', (data) => {
// 	const userList = document.getElementById('active-users');
// 	userList.innerHTML = data.users
// 		.map(
// 			(user) => `
//             <div class="user-item" onclick="insertPrivateMessage('${user}')">
//                 ${user} ${user === username ? '(you)' : ''}
//             </div>
//         `
// 		)
// 		.join('');
// });

function addMessage(user, message, type, sender, room) {
	if (!roomMessages[currentRoom]) {
		roomMessages[currentRoom] = [];
	}
	roomMessages[currentRoom].push({ user, message, type, sender });

	const chat = document.getElementById('chat');
	const messageDiv = document.createElement('div');
	messageDiv.className = `message ${type}`;
	// messageDiv.textContent = `${sender}: ${message}`;
	messageDiv.textContent = `${message}`;
	chat.appendChild(messageDiv)
	if (room != 'General'){
		const voiceButton = document.createElement('button')
		voiceButton.className = 'voiceButton'
		const voiceIcon = document.createElement('i')
		voiceIcon.className = 'fa fa-volume-up'
		if (sender == 'Bot'){
			voiceButton.style.backgroundColor = '#ecf0f1'
			voiceButton.style.color = 'black'
		}
		else{
			voiceButton.style.marginLeft = '1040px'
		}
		voiceButton.onclick = function() {createAudio(message, user, voiceIcon)}
		voiceButton.appendChild(voiceIcon)
		chat.appendChild(voiceButton)
	}
	chat.scrollTop = chat.scrollHeight;
}

let pendingVoiceIcon = null
function createAudio(message, user, voiceIcon){
	pendingVoiceIcon = voiceIcon; 
	voiceIcon.className = 'fa fa-spinner fa-spin';
	// console.log(user)
	socket.emit('createAudio',{
		msg: message,
		user: user
	})
}

let currentVoiceIcon = null
let audio = null

socket.on('readMessage', function(data) {
	if (audio && !audio.paused){
		audio.pause()
		audio.currentTime = 0
		currentVoiceIcon.className = 'fa fa-volume-up'
		return
	}
	const audioUrl = data.filepath + '?t=' + Date.now();
	audio = new Audio(audioUrl)
	currentAudio = audio
	if (currentVoiceIcon) { 
        currentVoiceIcon.className = 'fa fa-volume-up';
    }
	currentVoiceIcon = pendingVoiceIcon;
    currentVoiceIcon.className = 'fa fa-stop';
	audio.play()
	audio.onended = function() {
        currentVoiceIcon.className = 'fa fa-volume-up';
        currentVoiceIcon = null;
    };

})

function sendMessage() {
	const input = document.getElementById('message');
	const message = input.value.trim();

	if (!message) return; 

	socket.emit('message', {
			msg: message,
			room: currentRoom,
		});

	input.value = '';
	input.focus();
}

function joinRoom(room) {
	socket.emit('leave', { room: currentRoom });
	currentRoom = room;
	socket.emit('join', { room });

	highlightActiveRoom(room);

	const chat = document.getElementById('chat');
	chat.innerHTML = '';

	// if (roomMessages[room]) {
	// 	roomMessages[room].forEach((msg) => {
	// 		addMessage(msg.sender, msg.message, msg.type);
	// 	});
	// }
}

function handleKeyPress(event) {
	if (event.key === 'Enter' && !event.shiftKey) {
		event.preventDefault();
		sendMessage();
	}
}

let chat;
document.addEventListener('DOMContentLoaded', () => {
	chat = new ChatApp();
	if ('Notification' in window) {
		Notification.requestPermission();
	}
});

function highlightActiveRoom(room) {
	document.querySelectorAll('.room-item').forEach((item) => {
		item.classList.remove('active-room');
		if (item.textContent.trim() === room) {
			item.classList.add('active-room');
		}
	});
}

function createNewChat() {
    let room = document.getElementById("name").value;
    socket.emit('create_new_chat', {
		roomName: room
	});
    document.getElementById("dialog").close();
	// location.reload();
}
let currentName;
function openEditDialog(room){
	currentName = room
	document.getElementById('edit-dialog').showModal()
	// console.log(currentName)
}

function editRoomName(){
	let room = currentName
	let newName = document.getElementById("editting-name").value;
	socket.emit('edit-name',{
		room: room,
		newName: newName
	})
	location.reload()
}

function openDeleteDialog(room){
	currentName = room
	document.getElementById('delete-dialog').showModal()
	// console.log(currentName)
}

function deleteRoom(){
	socket.emit('delete',{room: currentName})
	location.reload()
}


socket.on("new_chat", function(data) {

    const roomList = document.getElementById("room-list");

	const container = document.createElement("div");
	container.className = "chat-contain"

    const div = document.createElement("div");
    div.className = "room-item";
    div.textContent = data.room;
	div.onclick = function () {
        joinRoom(data.room);
    };

	const editButton = document.createElement("button")
	editButton.className = "btn"
	editButton.onclick = function(){
		openEditDialog(data.room)
	}
	editButton.innerHTML = '<i class="fa fa-edit"></i>'

	const deleteButton = document.createElement("button")
	deleteButton.className = "btn"
	deleteButton.onclick = function(){
		deleteRoom(data.room)
	}
	deleteButton.innerHTML = '<i class="fa fa-trash"></i>'

    container.appendChild(div)
	container.appendChild(editButton)
	container.appendChild(deleteButton)

    roomList.appendChild(container);

	console.log("Đã nhận chat mới");
});

// socket.on("login_error", function(data){
// 	const errorMessage = document.getElementById("errorMessage")
// 	const content = document.createTextNode(data.msg)
// 	errorMessage.appendChild(content)
// })
