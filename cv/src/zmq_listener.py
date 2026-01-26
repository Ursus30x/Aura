import zmq
import json
import sys

def main():
    # Inicjalizacja kontekstu i gniazda ZeroMQ (Subscriber)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    # Połączenie z portem, na którym nadaje face_mesh_app.py
    # Jeśli uruchamiasz to na innej maszynie, zamień 'localhost' na IP tamtej maszyny
    print("Connecting to tcp://localhost:5555...")
    socket.connect("tcp://localhost:5555")
    
    # Subskrypcja wszystkich wiadomości (pusty filtr)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    print("Listening for Face Mesh data... (Press Ctrl+C to stop)")
    
    try:
        while True:
            # Odbiór wiadomości jako string
            message = socket.recv_string()
            
            # Parsowanie i wyświetlanie
            data = json.loads(message)
            print(f"Received: {json.dumps(data, indent=2)}")
            
    except KeyboardInterrupt:
        print("\nListener stopped.")

if __name__ == "__main__":
    main()