# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import json
from datetime import datetime, timedelta

from tornado import gen, web
from tornado.escape import json_encode, json_decode, url_escape
from tornado.httpclient import HTTPClient, AsyncHTTPClient, HTTPError
from tornado.ioloop import PeriodicCallback

from notebook.services.kernels.kernelmanager import MappingKernelManager
from notebook.services.sessions.sessionmanager import (
    SessionManager as BaseSessionManager
)
from jupyter_client.kernelspec import KernelSpecManager
from notebook.utils import url_path_join

from traitlets import Instance, Unicode, default

# TODO: Find a better way to specify global configuration options
# for a server extension.
DEFAULT_KG_SERVER_NAME = 'default'
DEFAULT_KG_KERNEL_NAME = os.getenv('DEFAULT_KG_KERNEL_NAME', 'python3')
DEFAULT_KG_IP = os.getenv('DEFAULT_KG_IP')
DEFAULT_KG_PORT = os.getenv('DEFAULT_KG_PORT')
KG_HEADERS = json.loads(os.getenv('KG_HEADERS', '{}'))
KG_HEADERS.update({
    'Authorization': 'token {}'.format(os.getenv('KG_AUTH_TOKEN', ''))
})
VALIDATE_KG_CERT = os.getenv('VALIDATE_KG_CERT') not in ['no', 'false']

KG_CLIENT_KEY = os.getenv('KG_CLIENT_KEY')
KG_CLIENT_CERT = os.getenv('KG_CLIENT_CERT')
KG_CLIENT_CA = os.getenv('KG_CLIENT_CA')

KG_HTTP_USER = os.getenv('KG_HTTP_USER')
KG_HTTP_PASS = os.getenv('KG_HTTP_PASS')

KG_CONNECT_TIMEOUT = float(os.getenv('KG_CONNECT_TIMEOUT', 20.0))
KG_REQUEST_TIMEOUT = float(os.getenv('KG_REQUEST_TIMEOUT', 20.0))

JUPYTERHUB_USER = os.getenv('JUPYTERHUB_USER')
JUPYTERHUB_API_URL = os.getenv('JUPYTERHUB_API_URL')
JUPYTERHUB_API_TOKEN = os.getenv('JUPYTERHUB_API_TOKEN')

KERNEL_SERVER_CPU_MAX_NUM = int(os.getenv('KERNEL_SERVER_CPU_MAX_NUM', 4))
KERNEL_SERVER_MEMORY_MAX_NUM = int(os.getenv('KERNEL_SERVER_MEMORY_MAX_NUM', 16))
KERNEL_SERVER_STORAGE_MAX_NUM = int(os.getenv('KERNEL_SERVER_STORAGE_MAX_NUM', 20))
KERNEL_SERVER_GPU_MAX_NUM = int(os.getenv('KERNEL_SERVER_GPU_MAX_NUM', 5))
KERNEL_SERVER_SPAWN_TIMEOUT = int(os.getenv('KERNEL_SERVER_SPAWN_TIMEOUT', 120))
KERNEL_SERVER_IMAGES = json.loads(os.getenv('KERNEL_SERVER_IMAGES', '{}'))


def load_connection_args(**kwargs):
    if KG_CLIENT_CERT:
        kwargs["client_key"] = kwargs.get("client_key", KG_CLIENT_KEY)
        kwargs["client_cert"] = kwargs.get("client_cert", KG_CLIENT_CERT)
        if KG_CLIENT_CA:
            kwargs["ca_certs"] = kwargs.get("ca_certs", KG_CLIENT_CA)
    kwargs['connect_timeout'] = kwargs.get('connect_timeout', KG_CONNECT_TIMEOUT)
    kwargs['request_timeout'] = kwargs.get('request_timeout', KG_REQUEST_TIMEOUT)
    kwargs['headers'] = kwargs.get('headers', KG_HEADERS)
    kwargs['validate_cert'] = kwargs.get('validate_cert', VALIDATE_KG_CERT)
    if KG_HTTP_USER:
        kwargs['auth_username'] = kwargs.get('auth_username', KG_HTTP_USER)
    if KG_HTTP_PASS:
        kwargs['auth_password'] = kwargs.get('auth_password', KG_HTTP_PASS)
    return kwargs


@gen.coroutine
def fetch_kg(gateway_url, **kwargs):
    """Make an async request to kernel gateway endpoint."""
    client = AsyncHTTPClient()

    kwargs = load_connection_args(**kwargs)

    response = yield client.fetch(gateway_url, **kwargs)
    raise gen.Return(response)


@gen.coroutine
def fetch_hub(entry, **kwargs):
    client = AsyncHTTPClient()

    kwargs.setdefault('headers', {}).update({
        'Authorization': 'token {}'.format(JUPYTERHUB_API_TOKEN)
    })
    url = url_path_join(JUPYTERHUB_API_URL, entry)
    response = yield client.fetch(url, **kwargs)
    raise gen.Return(response)


class RemoteKernelManager(MappingKernelManager):
    """Kernel manager that supports remote kernels hosted by Jupyter
    kernel gateway."""

    kernels_endpoint_env = 'KG_KERNELS_ENDPOINT'
    kernels_endpoint = Unicode(config=True,
        help="""The kernel gateway API endpoint for accessing kernel resources
        (KG_KERNELS_ENDPOINT env var)""")

    @default('kernels_endpoint')
    def kernels_endpoint_default(self):
        return os.getenv(self.kernels_endpoint_env, '/api/kernels')

    # TODO: The notebook code base assumes a sync operation to determine if
    # kernel manager has a kernel_id (existing kernel manager stores kernels
    # in dictionary).  Keeping such a dictionary in sync with remote KG is
    # NOT something we want to do, is it?
    #
    # options:
    #  - update internal dictionary on every /api/kernels request
    #  - replace `__contains__` with more formal async get_kernel() API
    #    (requires notebook code base changes)
    _kernels = {}

    def __contains__(self, kernel_id):
        self.log.debug('RemoteKernelManager.__contains__ {}'.format(kernel_id))
        return kernel_id in self._kernels

    def _remove_kernel(self, kernel_id):
        """Remove a kernel from our mapping, mainly so that a dead kernel can be
        removed without having to call shutdown_kernel.

        The kernel object is returned.

        Parameters
        ----------
        kernel_id: kernel UUID
        """
        try:
            return self._kernels.pop(kernel_id)
        except KeyError:
            pass

    def _kernel_id_to_url(self, kernel_id):
        """Builds a url for the given kernel UUID.

        Parameters
        ----------
        kernel_id: kernel UUID
        """
        server_name = self._kernels[kernel_id]['server_name']
        server = KernelServerManager.servers[server_name]
        host = 'http://{}:{}'.format(server['ip'], server['port'])
        return url_path_join(host, self.kernels_endpoint, url_escape(str(kernel_id)))

    @gen.coroutine
    def start_kernel(self, kernel_id=None, path=None, **kwargs):
        """Start a kernel for a session and return its kernel_id.

        Parameters
        ----------
        kernel_id : uuid
            The uuid to associate the new kernel with. If this
            is not None, this kernel will be persistent whenever it is
            requested.
        path : API path
            The API path (unicode, '/' delimited) for the cwd.
            Will be transformed to an OS path relative to root_dir.
        """
        self.log.info(
            'Request start kernel: kernel_id=%s, path="%s"',
            kernel_id, path
        )

        if kernel_id is None:
            server_name = kwargs.get('server_name', DEFAULT_KG_SERVER_NAME)
            if server_name is None and (not DEFAULT_KG_IP or not DEFAULT_KG_PORT):
                self.log.info('Local KG not exists, can not start new kernel')
                raise gen.Return(None)

            server = KernelServerManager.servers.get(server_name)
            if server is None:
                self.log.info('Gateway server not exists, can not start new kernel')
                raise gen.Return(None)

            kernel_name = kwargs['kernel_name'] if kwargs.get('kernel_name') else server['kernel_name']
            self.log.info("Request new kernel")
            kernel_env = {k: v for (k, v) in dict(os.environ).items() if k.startswith('KERNEL_') or
                    k in os.environ.get('KG_ENV_WHITELIST', '').split(",")}
            json_body = json_encode({'name': kernel_name, 'env': kernel_env})

            host = 'http://{}:{}'.format(server['ip'], server['port'])
            default_server_url = url_path_join(host, self.kernels_endpoint)
            response = yield fetch_kg(default_server_url, method='POST', body=json_body)
            kernel = json_decode(response.body)
            kernel['server_name'] = server_name
            kernel_id = kernel['id']
            self.log.info("Kernel started: {} at server {}".format(kernel_id, server_name))
        else:
            kernel = yield self.get_kernel(kernel_id)
            kernel_id = kernel['id']
            self.log.info("Using existing kernel: %s" % kernel_id)
        self._kernels[kernel_id] = kernel
        raise gen.Return(kernel_id)

    @gen.coroutine
    def get_kernel(self, kernel_id=None, **kwargs):
        """Get kernel for kernel_id.

        Parameters
        ----------
        kernel_id : uuid
            The uuid of the kernel.
        """
        if kernel_id not in self._kernels:
            yield self.list_kernels(**kwargs)

        if kernel_id not in self._kernels:
            raise gen.Return(None)

        kernel = self._kernels.get(kernel_id)
        raise gen.Return(kernel)

    @gen.coroutine
    def kernel_model(self, kernel_id):
        """Return a dictionary of kernel information described in the
        JSON standard model.

        Parameters
        ----------
        kernel_id : uuid
            The uuid of the kernel.
        """
        self.log.debug("RemoteKernelManager.kernel_model: %s", kernel_id)
        model = yield self.get_kernel(kernel_id)
        raise gen.Return(model)

    @gen.coroutine
    def list_kernels(self, **kwargs):
        """Get a list of kernels."""
        self.log.info("Request list kernels: %s", kwargs)
        self._kernels = {}
        for server_name, server in KernelServerManager.servers.items():
            host = 'http://{}:{}'.format(server['ip'], server['port'])
            response = yield fetch_kg(url_path_join(host, self.kernels_endpoint), method='GET')
            server_kernels = json_decode(response.body)
            for kernel in server_kernels:
                kernel['server_name'] = server_name
                self._kernels[kernel['id']] = kernel

        raise gen.Return(list(self._kernels.values()))

    @gen.coroutine
    def shutdown_kernel(self, kernel_id):
        """Shutdown a kernel by its kernel uuid.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel to shutdown.
        """
        if kernel_id not in self._kernels:
            yield self.list_kernels()

        if kernel_id not in self._kernels:
            raise gen.Return(None)

        self.log.info("Request shutdown kernel: %s", kernel_id)
        kernel_url = self._kernel_id_to_url(kernel_id)
        self.log.info("Request delete kernel at: %s", kernel_url)
        response = yield fetch_kg(kernel_url, method='DELETE')
        self.log.info("Shutdown kernel response: %d %s",
            response.code, response.reason)
        # self._remove_kernel(kernel_id)

    @gen.coroutine
    def restart_kernel(self, kernel_id, now=False, **kwargs):
        """Restart a kernel by its kernel uuid.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel to restart.
        """
        if kernel_id not in self._kernels:
            yield self.list_kernels(**kwargs)

        if kernel_id not in self._kernels:
            raise gen.Return(None)

        self.log.info("Request restart kernel: %s", kernel_id)
        kernel_url = self._kernel_id_to_url(kernel_id) + '/restart'
        self.log.info("Request restart kernel at: %s", kernel_url)
        response = yield fetch_kg(kernel_url, method='POST', body=json_encode({}))
        self.log.info("Restart kernel response: %d %s", response.code, response.reason)

    @gen.coroutine
    def interrupt_kernel(self, kernel_id, **kwargs):
        """Interrupt a kernel by its kernel uuid.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel to interrupt.
        """
        if kernel_id not in self._kernels:
            yield self.list_kernels(**kwargs)

        if kernel_id not in self._kernels:
            raise gen.Return(None)

        self.log.info("Request interrupt kernel: %s", kernel_id)
        kernel_url = self._kernel_id_to_url(kernel_id) + '/interrupt'
        self.log.info("Request interrupt kernel at: %s", kernel_url)
        try:
            response = yield fetch_kg(kernel_url, method='POST', body=json_encode({}))
        except HTTPError as error:
            if error.code == 404:
                pass
        else:
            self.log.info("Interrupt kernel response: %d %s", response.code, response.reason)

    def shutdown_all(self):
        """Shutdown all kernels."""
        # Note: We have to make this sync because the NotebookApp does not wait for async.
        kwargs = {'method': 'DELETE'}
        kwargs = load_connection_args(**kwargs)
        client = HTTPClient()
        for kernel_id in self._kernels.keys():
            kernel_url = self._kernel_id_to_url(kernel_id)
            self.log.info("Request delete kernel at: %s", kernel_url)
            try:
                response = client.fetch(kernel_url, **kwargs)
            except HTTPError:
                pass
            self.log.info("Delete kernel response: %d %s",
                response.code, response.reason)
        client.close()

    @classmethod
    def get_kernel_server(cls, kernel_id):
        server_name = cls._kernels[kernel_id]['server_name']
        return KernelServerManager.servers[server_name]


class RemoteKernelSpecManager(KernelSpecManager):
    kernelspecs_endpoint_env = 'KG_KERNELSPECS_ENDPOINT'
    kernelspecs_endpoint = Unicode(config=True,
        help="""The kernel gateway API endpoint for accessing kernelspecs
        (KG_KERNELSPECS_ENDPOINT env var)""")

    @default('kernelspecs_endpoint')
    def kernelspecs_endpoint_default(self):
        return os.getenv(self.kernelspecs_endpoint_env, '/api/kernelspecs')

    @gen.coroutine
    def list_kernel_specs(self):
        """Get a list of kernel specs."""
        self.log.info("Request list kernel specs at: %s", self.kernelspecs_endpoint)

        # only for default kernel specs
        if not DEFAULT_KG_IP or not DEFAULT_KG_PORT:
            raise gen.Return({
                'default': '',
                'kernelspecs': {},
            })

        default_server_url = url_path_join('http://{}:{}'.format(DEFAULT_KG_IP, DEFAULT_KG_PORT), self.kernelspecs_endpoint)
        response = yield fetch_kg(default_server_url, method='GET')
        kernel_specs = json_decode(response.body)
        raise gen.Return(kernel_specs)

    @gen.coroutine
    def get_kernel_spec(self, kernel_name, **kwargs):
        """Get kernel spec for kernel_name.

        Parameters
        ----------
        kernel_name : str
            The name of the kernel.
        """
        # only for default kernel specs
        if not DEFAULT_KG_IP or not DEFAULT_KG_PORT:
            raise gen.Return(None)

        kernel_spec_url = url_path_join('http://{}:{}'.format(DEFAULT_KG_IP, DEFAULT_KG_PORT), self.kernelspecs_endpoint, str(kernel_name))
        self.log.info("Request kernel spec at: %s" % kernel_spec_url)
        try:
            response = yield fetch_kg(kernel_spec_url, method='GET')
        except HTTPError as error:
            if error.code == 404:
                self.log.info("Kernel spec not found at: %s" % kernel_spec_url)
                kernel_spec = None
            else:
                raise
        else:
            kernel_spec = json_decode(response.body)
        raise gen.Return(kernel_spec)


class SessionManager(BaseSessionManager):
    kernel_manager = Instance('nb2kg.managers.RemoteKernelManager')

    @gen.coroutine
    def create_session(self, path=None, name=None, type=None,
                       kernel_name=None, kernel_id=None):
        """Creates a session and returns its model.

        Overrides base class method to turn into an async operation.
        """
        session_id = self.new_session_id()

        kernel = None
        if kernel_id is not None:
            # This is now an async operation
            kernel = yield self.kernel_manager.get_kernel(kernel_id)

        if kernel is not None:
            pass
        else:
            kernel_id = yield self.start_kernel_for_session(
                session_id, path, name, type, kernel_name,
            )

        result = yield self.save_session(
            session_id, path=path, name=name, type=type, kernel_id=kernel_id,
        )
        raise gen.Return(result)

    @gen.coroutine
    def save_session(self, session_id, path=None, name=None, type=None,
                     kernel_id=None):
        """Saves the items for the session with the given session_id

        Given a session_id (and any other of the arguments), this method
        creates a row in the sqlite session database that holds the information
        for a session.

        Parameters
        ----------
        session_id : str
            uuid for the session; this method must be given a session_id
        path : str
            the path for the given notebook
        kernel_id : str
            a uuid for the kernel associated with this session

        Returns
        -------
        model : dict
            a dictionary of the session model
        """
        # This is now an async operation
        session = yield super(SessionManager, self).save_session(
            session_id, path=path, name=name, type=type, kernel_id=kernel_id
        )
        raise gen.Return(session)

    @gen.coroutine
    def get_session(self, **kwargs):
        """Returns the model for a particular session.

        Takes a keyword argument and searches for the value in the session
        database, then returns the rest of the session's info.

        Overrides base class method to turn into an async operation.

        Parameters
        ----------
        **kwargs : keyword argument
            must be given one of the keywords and values from the session database
            (i.e. session_id, path, kernel_id)

        Returns
        -------
        model : dict
            returns a dictionary that includes all the information from the
            session described by the kwarg.
        """
        # This is now an async operation
        session = yield super(SessionManager, self).get_session(**kwargs)
        raise gen.Return(session)

    @gen.coroutine
    def update_session(self, session_id, **kwargs):
        """Updates the values in the session database.

        Changes the values of the session with the given session_id
        with the values from the keyword arguments.

        Overrides base class method to turn into an async operation.

        Parameters
        ----------
        session_id : str
            a uuid that identifies a session in the sqlite3 database
        **kwargs : str
            the key must correspond to a column title in session database,
            and the value replaces the current value in the session
            with session_id.
        """
        # This is now an async operation
        yield self.get_session(session_id=session_id)

        if not kwargs:
            # no changes
            return

        sets = []
        for column in kwargs.keys():
            if column not in self._columns:
                raise TypeError("No such column: %r" % column)
            sets.append("%s=?" % column)
        query = "UPDATE session SET %s WHERE session_id=?" % (', '.join(sets))
        self.cursor.execute(query, list(kwargs.values()) + [session_id])

    @gen.coroutine
    def row_to_model(self, row):
        """Takes sqlite database session row and turns it into a dictionary.

        Overrides base class method to turn into an async operation.
        """
        # Retrieve kernel for session, which is now an async operation
        kernel = yield self.kernel_manager.get_kernel(row['kernel_id'])
        if kernel is None:
            # The kernel was killed or died without deleting the session.
            # We can't use delete_session here because that tries to find
            # and shut down the kernel.
            self.cursor.execute("DELETE FROM session WHERE session_id=?",
                                (row['session_id'],))
            raise KeyError

        model = {
            'id': row['session_id'],
            'path': row['path'],
            'name': row['name'],
            'type': row['type'],
            'kernel': kernel
        }
        if row['type'] == 'notebook':  # Provide the deprecated API.
            model['notebook'] = {'path': row['path'], 'name': row['name']}

        raise gen.Return(model)

    @gen.coroutine
    def list_sessions(self):
        """Returns a list of dictionaries containing all the information from
        the session database.

        Overrides base class method to turn into an async operation.
        """
        c = self.cursor.execute("SELECT * FROM session")
        result = []
        # We need to use fetchall() here, because row_to_model can delete rows,
        # which messes up the cursor if we're iterating over rows.
        for row in c.fetchall():
            try:
                # This is now an async operation
                model = yield self.row_to_model(row)
                result.append(model)
            except KeyError:
                pass
        raise gen.Return(result)

    @gen.coroutine
    def delete_session(self, session_id):
        """Deletes the row in the session database with given session_id.

        Overrides base class method to turn into an async operation.
        """
        # This is now an async operation
        session = yield self.get_session(session_id=session_id)
        yield gen.maybe_future(self.kernel_manager.shutdown_kernel(session['kernel']['id']))
        self.cursor.execute("DELETE FROM session WHERE session_id=?", (session_id,))


class KernelServerManager(object):
    servers = {}

    def __init__(self):
        self._poll_fail_time = 0
        self._poll_callback = None
        self._server_name = None

    @gen.coroutine
    def create_server(self, model):
        self._check_params(model)

        curr_time = datetime.now()
        self._server_name = server_name = self._format_server_name(model)
        if server_name in self.servers:
            if self.servers[server_name]['is_active']:
                raise web.HTTPError(422, 'Same kernel server already exists!')
            if self.servers[server_name]['spawn_time'] + timedelta(seconds=KERNEL_SERVER_SPAWN_TIMEOUT) > curr_time:
                raise web.HTTPError(422, 'Same kernel server being spawned!')

        server_image = self._server_image(model)
        if not server_image:
            raise web.HTTPError(422, '{} image not found'.format(model['kernel_name']))

        entry = 'users/{}/servers/{}'.format(JUPYTERHUB_USER, url_escape(server_name))
        params = {
            'kubespawner_override': {
                'singleuser_image_spec': server_image,
                'cpu_limit': '{}m'.format(model['cpu']),
                'mem_limit': '{}M'.format(model['memory']),
            }
        }
        if model.get('gpu', 0) > 0:
            params['kubespawner_override']['extra_resource_guarantees'] = {"nvidia.com/gpu": str(model['gpu'])}
            params['kubespawner_override']['extra_resource_limits'] = {"nvidia.com/gpu": str(model['gpu'])}

        json_body = json_encode(params)
        response = yield fetch_hub(entry, method='POST', body=json_body)
        if response.code not in (201, 202):
            self.log.info("Create kernel server failed, response: %d %s", response.code, response.reason)
            raise web.HTTPError(response.code, 'Create kernel server failed', reason=response.reason)

        self._start_polling()

        server = json_decode(response.body)
        self.servers[server_name] = model
        self.servers[server_name].update({
            'ip': server['ip'],
            'port': server['port'],
            'pending': 'spawn',
            'ready': False,
            'spawn_time': curr_time,
        })
        raise gen.Return({
            'server_name': server_name,
            'is_active': self._is_active(self.servers[server_name]),
        })

    @classmethod
    @gen.coroutine
    def get_server(cls, server_name):
        if server_name not in cls.servers:
            raise web.HTTPError(404, '{} server not found'.format(server_name))

        raise gen.Return({
            'is_active': cls._is_active(cls.servers[server_name]),
        })

    @classmethod
    def _is_active(self, server):
        return server['ready']

    def _stop_polling(self):
        """Stop polling for kernel server's running state"""
        if self._poll_callback:
            self._poll_callback.stop()
            self._poll_callback = None

    def _start_polling(self):
        """Start polling periodically for kernel server's running state"""
        poll_interval = 10
        self.log.debug("Polling subprocess every %is", poll_interval)

        self._stop_polling()

        self._poll_callback = PeriodicCallback(
            self._poll_and_notify,
            1e3 * poll_interval
        )
        self._poll_callback.start()

    @gen.coroutine
    def _poll_and_notify(self):
        """Used as a callback to periodically poll the process and notify any watchers"""
        server = yield self._poll()
        # not exists, or has stoped
        if not server['pending'] and not server['ready']:
            self._poll_fail_time += 1
            if self._poll_fail_time > 1:
                # Two times continuous fail
                self._stop_polling()
        else:
            self._poll_fail_time = 0

        self.servers[self._server_name].update({
            'pending': server['pending'],
            'ready': server['ready'],
        })

    @gen.coroutine
    def _poll(self):
        entry = 'users/{}/servers/{}'.format(JUPYTERHUB_USER, url_escape(self._server_name))
        response = yield fetch_hub(entry, method='GET')
        if response.code != 200:
            self.log.info("Get kernel server failed, response: %d %s", response.code, response.reason)
            server = {
                'pending': '',
                'ready': False,
            }
        else:
            server = json_decode(response.body)
        return server

    def _format_server_name(self, model):
        return '_'.join(model['kernel_name'], str(model['cpu']), str(model['memory']), str(model['storage']), str(model.get('gpu', 0)))

    def _server_image(self, model):
        return KERNEL_SERVER_IMAGES.get('{}_{}'.format(model['kernel_name'], model.get('framework', 'TensorFlow')))

    def _check_params(self, model):
        if model is None:
            raise web.HTTPError(422, 'No JSON data provided')

        if not model['kernel_name']:
            raise web.HTTPError(422, 'Missing field in JSON data: kernel_name')

        if not model['cpu'] or model['cpu'] > KERNEL_SERVER_CPU_MAX_NUM:
            raise web.HTTPError(422, 'Invalid cpu num')

        if not model['memory'] or model['memory'] > KERNEL_SERVER_MEMORY_MAX_NUM:
            raise web.HTTPError(422, 'Invalid memory num')

        if not model['storage'] or model['storage'] > KERNEL_SERVER_STORAGE_MAX_NUM:
            raise web.HTTPError(422, 'Invalid storage num')

        if model.get('gpu', 0) > KERNEL_SERVER_GPU_MAX_NUM:
            raise web.HTTPError(422, 'Invalid gpu num')


if DEFAULT_KG_IP and DEFAULT_KG_PORT:
    KernelServerManager.servers[DEFAULT_KG_SERVER_NAME] = {
        'kernel_name': DEFAULT_KG_KERNEL_NAME,
        'ip': DEFAULT_KG_IP,
        'port': DEFAULT_KG_PORT,
    }
