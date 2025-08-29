import asyncio
from bleak import BleakClient, BleakScanner
import threading
import time
# These UUIDs must match the ones in your Arduino sketch
DEVICE_NAME = "StepperMotorBoard1"
COMMAND_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"
# The status characteristic is not used in this script but is defined for completeness
# STATUS_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"



class BLE_Client:
    """
    This is a generalised BLE client class that can be used to connect to a BLE device acting as a central. 
    We are bridging the async bleak package with a synchronous interface to make it easier to use in a blocking manner.
    If we know the device command UUIDs we can broadcast commands to the device.
    The device must be running a compatible Arduino sketch that listens for these commands with the same UUIDs.

    The example device name and command UUID are for a stepper motor arduino project
    """
    def __init__(self, device_name=DEVICE_NAME, command_uuid=COMMAND_UUID, timeout=3):
        self.device_name = device_name
        self.command_uuid = command_uuid
        self.timeout = timeout
        self.client = None
        self.loop = None
        self.thread = None
        self.connected = False
        self.device_info = None


    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    def handshake(self):
        """Establish a connection to the device with a bridge between
        async and sync operation."""
        # Start the async event loop in a separate thread
        self.thread = threading.Thread(target=self.__run_async_loop__, daemon=True)
        self.thread.start()
        
        # Wait for the loop to start
        time.sleep(0.1)
        # Connect to the device
        return self.__run_coroutine__(self.__async_connect__())
        # time.sleep(self.timeout)


    def __run_async_loop__(self):
        """Run the async event loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def __run_coroutine__(self, coro):
        """Run a coroutine in the async loop and return the result synchronously."""
        if self.loop is None:
            raise RuntimeError("Event loop not started. Call connect() first.")
        
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=self.timeout)
    
    async def __async_connect__(self):
        """Asynchronous connection method."""
        print("Scanning for devices...")
        device = await BleakScanner.find_device_by_name(self.device_name)
        
        if device is None:
            raise ConnectionError(f"Could not find a device named '{self.device_name}'")
        
        print(f"Connecting to {device.name} ({device.address})...")
        
        self.client = BleakClient(device, timeout=self.timeout)
        await self.client.connect()
        
        if not self.client.is_connected:
            raise ConnectionError(f"Failed to connect to {device.address}")
        
        print(f"Connected to {self.client.address}")
        self.connected = True
        return self.connected
    
    async def __async_disconnect__(self):
        """Asynchronous disconnect method."""
        if self.client and self.client.is_connected:
            await self.client.disconnect()
        self.connected = False
    
    async def __async_send_command__(self, command):
        """Asynchronous command sending method."""
        if not self.client or not self.client.is_connected:
            raise RuntimeError("Not connected to device")
        
        await self.client.write_gatt_char(self.command_uuid, bytearray(command.encode()))
        return True
    

    def disconnect(self):
        """Disconnect from the device and clean up."""
        if self.loop and not self.loop.is_closed():
            self.__run_coroutine__(self.__async_disconnect__())
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        
        print("Disconnected from the device.")
    
    def send_command(self, command):
        """Send a command to the stepper motor."""
        if not self.connected:
            raise RuntimeError("Not connected to device. Call connect() first.")
        
        self.__run_coroutine__(self.__async_send_command__(command))
        #print(f"Sent: '{command}'")
        return True
    
    def is_connected(self):
        """Check if the device is connected."""
        return self.connected and self.client and self.client.is_connected

    def get_device_info(self):
        """Retrieve information about the connected device."""
        if not self.is_connected():
            raise RuntimeError("Not connected to device.")
        return {
            "name": self.client.name,
            "address": self.client.address,
            "is_connected": self.client.is_connected
        }


if __name__ == '__main__':
    with BLE_Client() as bc:
        print('\n\n################# ENTERING BLE_Client AS  >>>>> bc <<<<<< ####################\n\n')
        import code; code.interact(local=locals())
        bc.handshake()
        # Example usage