using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Concurrent;

public class UDPReceiver : MonoBehaviour
{
    [Header("Network Settings")]
    public int port = 5555;
    
    private UdpClient udpClient;
    private Thread receiveThread;
    private bool isRunning;

    // Thread-safe queue to pass JSON packets to the main Unity thread
    public ConcurrentQueue<string> packetQueue = new ConcurrentQueue<string>();

    void Start()
    {
        StartReceiving();
    }

    private void StartReceiving()
    {
        isRunning = true;
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true; // Prevents blocking Unity shutdown
        receiveThread.Start();
        Debug.Log($"[UDPReceiver] Listening on port {port}");
    }

    private void ReceiveData()
    {
        try 
        {
            udpClient = new UdpClient(port);
            IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);

            while (isRunning)
            {
                byte[] data = udpClient.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);
                
                // Keep the queue small to prevent memory leaks if Unity lags
                if (packetQueue.Count > 10) {
                    packetQueue.TryDequeue(out _); 
                }
                
                packetQueue.Enqueue(text);
            }
        }
        catch (System.Exception e)
        {
            if (isRunning)
                Debug.LogWarning("[UDPReceiver] Error: " + e.Message);
        }
    }

    void OnDestroy()
    {
        isRunning = false;
        if (udpClient != null)
        {
            udpClient.Close();
        }
        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(500);
        }
    }
}