import zmq
import threading 
import numpy as np
import time

class TesolloJointStateReceiver:
    '''
    Applies to any ZMQ-broadcasted mocap data with fixed shape (21,3) and dtype float32.
    Runs a background thread to continuously receive and update latest data.
    '''
    def __init__(self, port=8002):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(
            zmq.CONFLATE, 1
        )  # Keep only the latest message
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics

        self.socket.connect(f"tcp://localhost:{port}")

        self._latest_pos = np.zeros(20)

        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                message = self.socket.recv_json()
                data = message["q"]

                # print(data)

                with self._lock:
                    self._latest_pos = np.array(data)
            except zmq.Again:
                time.sleep(0.001)

    def get(self):
        with self._lock:
            return list(self._latest_pos)
       
    def close(self):
        self._running = False
        self._thread.join()
        self.socket.close()