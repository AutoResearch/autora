STRING_FORMAT = 'utf-8'
TIMEOUT = 5

# protocol keywords
OK = b'OK'
TRANSFER_SESSION_ID = b'SESSION_ID'
NEGOTIATE_STATUS = b'NEGOTIATE'
TRANSFER_CLIENT_STATUS = b'TRANSFER_CLIENT_STATUS'
TRANSFER_SERVER_STATUS = b'TRANSFER_SERVER_STATUS'
INITIATE_TRANSFER = b'TRANSFER'
TRANSFER_FILEPATH = b'FILEPATH'
TRANSFER_DATA = b'FILEDATA'
TRANSFER_COMPLETE = b'FILECOMPLETE'
REQUEST_COMPLETE = b'REQUESTCOMPLETE'
START_EXPERIMENT = b'STARTEXP'
CLEAR_SESSIONS = b'CLEAR_SESSIONS'

# server/tinkerforge status

STATUS_ERROR = -1
STATUS_NEGOTIATING_STATUS = 0
STATUS_CONNECTED = 1
STATUS_INITIATED_TRANSFER = 3
STATUS_RECEIVING_FILE_PATH = 4
STATUS_RECEIVING_FILE_DATA = 5
STATUS_SENDING_FILE_PATH = 6
STATUS_SENDING_FILE_DATA = 7
STATUS_COMPLETED_TRANSFER = 8
STATUS_TRANSFERRING_SESSION_ID = 9
STATUS_RECEIVED_SESSION_ID = 10
STATUS_REQUESTING_EXP = 11
STATUS_LISTENING = 12
STATUS_ABORT = 13
STATUS_RUNNING_EXP = 14
STATUS_TERMINATING_SESSION = 15
STATUS_CLEARING_SESSIONS = 16

# process
JOB_STATUS_CONNECTED = STATUS_CONNECTED + 20
JOB_STATUS_SENT_EXPERIMENT_FILE = 22
JOB_STATUS_SENT_SEQUENCE_FILE = 23
JOB_STATUS_REQUESTED_EXPERIMENT = 24
JOB_STATUS_COMPLETED_EXPERIMENT = 25
JOB_STATUS_RECEIVED_DATA_FILE = 26
JOB_STATUS_COMPLETED_JOB = 27