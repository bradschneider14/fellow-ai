import pika
import os
import json
import traceback


class BaseRMQWorker:
    '''
    A base implementation of an RMQ worker that handles connection, 
    message consumption, and publishing. Subclasses should implement process_message.

    Set RMQ_HOST environment variable to the hostname of the rabbitmq server (default: localhost)
    '''

    def __init__(self, listen_queue: str, publish_queue: str = None):
        '''
        Initialize the worker.

        Args    :
            listen_queue (str): The name of the queue to listen to.
            publish_queue (str, optional): The name of the queue to publish to. Defaults to None.
        '''
        self.listen_queue = listen_queue
        self.publish_queue = publish_queue
        self.connection = None
        self.channel = None

    def connect(self):
        '''
        Create a connection to the rabbitmq server, declare the listen and publish queues.

        Must be called before start_listening.
        '''

        # Default to localhost if RMQ_HOST is not set
        host = os.environ.get("RMQ_HOST", "localhost")
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host)
        )
        self.channel = self.connection.channel()

        # Declare the queues to ensure they exist (durable=True so they survive rabbitmq restarts)
        self.channel.queue_declare(queue=self.listen_queue, durable=True)
        if self.publish_queue:
            self.channel.queue_declare(queue=self.publish_queue, durable=True)

    def publish_output_to_queue(self, payload: dict, queue_name: str = None):
        '''
        Publish a payload to a queue.

        Args:
            payload (dict): The payload to publish.
            queue_name (str, optional): The name of the queue to publish to. Defaults to the current publish_queue.
        '''

        if not self.channel:
            raise Exception("Cannot publish without an active connection.")

        target_queue = queue_name or self.publish_queue
        if not target_queue:
            raise ValueError("No target queue specified for publishing.")

        self.channel.basic_publish(
            exchange='',
            routing_key=target_queue,
            body=json.dumps(payload),
            properties=pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE  # Make message persistent
            )
        )
        print(f"[{self.__class__.__name__}] Published to {target_queue}: {payload}")

    def get_listening_queue(self) -> str:
        return self.listen_queue

    def start_listening(self):
        '''
        Start listening for messages on the listen queue. Blocks until an interrupt signal is received.
        '''

        self.connect()
        # Prefetch 1 message at a time to distribute workload
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.listen_queue,
            on_message_callback=self._internal_callback
        )
        print(
            f"[{self.__class__.__name__}] Waiting for messages on {self.listen_queue}. To exit press CTRL+C")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("Stopping...")
            self.connection.close()

    def _internal_callback(self, ch, method, properties, body):
        try:
            payload = json.loads(body.decode('utf-8'))
            print(
                f"[{self.__class__.__name__}] Received payload on {self.listen_queue}: {payload}")

            # Delegate to subclass
            self.process_message(payload)

            # Only ACK after successful processing
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error processing message: {str(e)}")
            traceback.print_exc()
            # NACK the message. requeue=False avoids infinite loops on poison messages.
            # In a production environment, a dead-letter exchange should be configured.
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def process_message(self, payload: dict):
        '''
        Process a message. Automatically called by the internal callback. Implementers should explicitly call publish_output_to_queue when they are ready to pass the message to the next worker - this is not called automatically.

        Args:
            payload (dict): The payload to process.
        '''
        raise NotImplementedError("Subclasses must implement process_message")
