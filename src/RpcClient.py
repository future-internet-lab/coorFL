import time
import pickle
import pika
import src.Log


class RpcClient:
    def __init__(self, client_id, model, address, username, password, train_func, device):
        self.model = model
        self.client_id = client_id
        self.address = address
        self.username = username
        self.password = password
        self.train_func = train_func
        self.device = device

        self.channel = None
        self.connection = None
        self.response = None

        self.connect()

    def wait_response(self):
        status = True
        while status:
            credentials = pika.PlainCredentials(self.username, self.password)
            reply_connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, '/', credentials))
            reply_channel = reply_connection.channel()
            reply_queue_name = f'reply_{self.client_id}'
            reply_channel.queue_declare(reply_queue_name, durable=False)
            method_frame, header_frame, body = self.channel.basic_get(queue=reply_queue_name, auto_ack=True)
            if body:
                status = self.response_message(body)
            time.sleep(0.5)

    def response_message(self, body):
        self.response = pickle.loads(body)
        src.Log.print_with_color(f"[<<<] Client received: {self.response['message']}", "blue")
        action = self.response["action"]
        state_dict = self.response["parameters"]

        if action == "START":
            # Read parameters and load to model
            if state_dict:
                if self.device != "cpu":
                    for key in state_dict:
                        state_dict[key] = state_dict[key].to(self.device)
                self.model.load_state_dict(state_dict)

            batch_size = self.response["batch_size"]
            lr = self.response["lr"]
            momentum = self.response["momentum"]
            label_counts = self.response["label_counts"]

            # Start training
            self.train_func(label_counts, batch_size, lr, momentum)

            # Stop training, then send parameters to server
            model_state_dict = self.model.state_dict()
            if self.device != "cpu":
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to('cpu')
            data = {"action": "UPDATE", "client_id": self.client_id,
                    "message": "Sent parameters to Server", "parameters": model_state_dict}
            src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
            self.send_to_server(data)
            return True
        elif action == "STOP":
            return False

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, '/', credentials))
        self.channel = self.connection.channel()

    def send_to_server(self, message):
        self.connect()
        self.response = None

        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

        return self.response
