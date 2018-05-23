# HTTP API

## Start remote kernel server

POST /api/kernel-servers/

Body:

```
{
    "kernel_name": "Python3"
    "framework": "TensorFlow" // TensorFlow, PyTorch, MxNet, Keras, default: TensorFlow
    "cpu": 1000,              // unit is m，100m=0.1CPU
    "memory": 1024,           // unit is MiB
    "storage": 10240,         // unit is MiB
    "gpu": 0                  // unit is gpu count
}
```

Result:

If successful, the server will return http\_statuse=201 and data, otherwise http\_status=422.

```
{
    "server_name": "Python3_1000_1024_10240_0",
    "is_active": false  // true: ready to use, false: being waiting
}
```

## Get kernel server

GET /api/kernel-servers/{server_name}

Result:

```
{
    "is_active": false  // true: ready to use, false: being waiting
}
```

## Start kernel

POST /api/kernel-servers/{server_name}/kernels

Body:

```
{
    "kernel_name": "Python3"  // Optional, select the default kernel if it is empty
}
```

Result:

If successful, the server will return http\_statuse=201 and data, otherwise http\_status=422.

```
{
    "id": "0f64ce38-8f6d-42ca-b251-1dade43baaf9",
    "name": "python3",
    "last_activity": "2018-05-30T05:42:49.728652Z",
    "execution_state": "idle",
    "connections": 1
}
```
