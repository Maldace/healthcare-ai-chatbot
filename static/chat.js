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
		data.username === username ? 'own' : 'other'
	);
});

socket.on('status', (data) => {
	addMessage('System', data.msg, 'system');
});

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

function addMessage(sender, message, type) {
	if (!roomMessages[currentRoom]) {
		roomMessages[currentRoom] = [];
	}
	roomMessages[currentRoom].push({ sender, message, type });

	const chat = document.getElementById('chat');
	const messageDiv = document.createElement('div');
	messageDiv.className = `message ${type}`;
	messageDiv.textContent = `${sender}: ${message}`;

	chat.appendChild(messageDiv);
	chat.scrollTop = chat.scrollHeight;
}

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

function deleteRoom(room){
	socket.emit('delete',{room: room})
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
// function login(){
// 	let username = document.getElementById("username").value
// 	let password = document.getElementById("password").value
// 	socket.emit('login',{
// 		username: username,
// 		password: password
// 	}, function(response) { // Gửi thông tin login lên Flask và nhận response
//     if (response.success) { // Kiểm tra đăng nhập thành công
//         window.location.href = "/home"; // Chuyển trình duyệt đến /home
//     }
// 	})
// }
