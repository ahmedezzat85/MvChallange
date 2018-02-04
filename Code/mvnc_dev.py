try:
    from mvnc import mvncapi as mvnc
    class MvNCS(object):
        """ A simple wrapper for the Movidius NCS device.
        """
        def __init__(self, dev_idx=0):
            # configure the NCS
            mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 1)

            # Get a list of ALL the sticks that are plugged in
            devices = mvnc.EnumerateDevices()
            if len(devices) == 0:
                raise ValueError('No devices found')

            # Pick the first stick to run the network
            self.dev = mvnc.Device(devices[dev_idx])

            # Open the NCS
            try:
                self.dev.OpenDevice()
            except:
                raise ValueError('Cannot Open NCS Device')

        def load_model(self, graph_file):
            with open(graph_file, mode='rb') as f:
                blob = f.read()

            self.ncs_graph = self.dev.AllocateGraph(blob)
            self.ncs_graph.SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)

        def forward(self, input_tensor):
            self.ncs_graph.LoadTensor(input_tensor, None)
            net_out, _ = self.ncs_graph.GetResult()
            return net_out

        def unload_model(self):
            self.ncs_graph.DeallocateGraph()

        def close(self):
            self.dev.CloseDevice()

except ImportError:
    class MvNCS(object):
        """ A simple wrapper for the Movidius NCS device.
        """
        def __init__(self, dev_idx=0):
            raise NotImplementedError('Movidius NCSDK is not installed')

