using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;

public class NetworkManagerUDP : MonoBehaviour
{
    [Header("Network Settings")]
    public int port = 5555;
    
    private UdpClient udpClient;
    private Thread receiveThread;
    private bool isRunning;

    private List<string> packetQueue = new List<string>();
    private readonly object queueLock = new object();

    void Start()
    {
        StartReceiving();
    }

    private void StartReceiving()
    {
        isRunning = true;
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
        Debug.Log("[UDPReceiver] Listening on port " + port);
    }

    private void ReceiveData()
    {
        try 
        {
            udpClient = new UdpClient(port);
            // Allow multiple applications to bind to the same port (useful during crashes/restarts)
            udpClient.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
            IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);

            while (isRunning)
            {
                // Check if data is available before blocking, or just block and handle the exception on Close
                if (udpClient.Available > 0)
                {
                    byte[] data = udpClient.Receive(ref anyIP);
                    string text = Encoding.UTF8.GetString(data);
                    
                    lock (queueLock)
                    {
                        if (packetQueue.Count > 10) packetQueue.RemoveAt(0);
                        packetQueue.Add(text);

                        //Debug.Log(text);
                    }
                }
                else
                {
                    Thread.Sleep(10); // Don't burn CPU if no data
                }
            }
        }
        catch (SocketException e)
        {
            // This is expected when the socket is closed while waiting for data
            if (isRunning) Debug.LogWarning("[UDPReceiver] SocketException: " + e.Message);
        }
        catch (Exception e)
        {
            if (isRunning) Debug.LogWarning("[UDPReceiver] Error: " + e.Message);
        }
    }

    public string GetNextPacket()
    {
        lock (queueLock)
        {
            if (packetQueue.Count > 0)
            {
                string p = packetQueue[0];
                packetQueue.RemoveAt(0);
                return p;
            }
        }
        return null;
    }

    void OnDisable()
    {
        Cleanup();
    }

    void OnDestroy()
    {
        Cleanup();
    }

    private void Cleanup()
    {
        isRunning = false;
        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
        }

        if (receiveThread != null)
        {
            if (receiveThread.IsAlive)
            {
                // Use Join with timeout instead of Abort (Abort is deprecated/unsafe)
                receiveThread.Join(100); 
            }
            receiveThread = null;
        }
    }
}
