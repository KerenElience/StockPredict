import logging, os
import time
from logging import FileHandler
from logging.handlers import RotatingFileHandler

root = os.path.dirname(os.path.dirname(__file__))

class SafeFileHandler(FileHandler):
    def __init__(self, filename, mode: str= 'a', encoding: str=None, *args, **kwargs):
        """
        Use the specified filename for streamed logging
        """
        FileHandler.__init__(self, filename, mode, encoding, *args, **kwargs)
        self.suffix = "%Y-%m-%d"
        self.suffix_time = ""

    def emit(self, record):
        """
        Emit a record.

        Always check time 
        """
        try:
            if self.check_baseFilename(record):
                self.build_baseFilename()
            FileHandler.emit(self, record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def check_baseFilename(self, record):
        """
        Determine if builder should occur.

        record is not used, as we are just comparing times, 
        but it is needed so the method signatures are the same
        """
        timeTuple = time.localtime()

        if self.suffix_time != time.strftime(self.suffix, timeTuple) or not os.path.exists(self.baseFilename+'.'+self.suffix_time):
            return 1
        else:
            return 0
        
    def build_baseFilename(self):
        """
        do builder; in this case, 
        old time stamp is removed from filename and
        a new time stamp is append to the filename
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        # remove old suffix
        if self.suffix_time != "":
            index = self.baseFilename.find("."+self.suffix_time)
            if index == -1:
                index = self.baseFilename.rfind(".")
            self.baseFilename = self.baseFilename[:index]

        # add new suffix
        currentTimeTuple = time.localtime()
        self.suffix_time = time.strftime(self.suffix, currentTimeTuple)
        self.baseFilename  = self.baseFilename + "." + self.suffix_time

        self.mode = 'a'
        if not self.delay:
            self.stream = self._open()

logpath = os.path.join(root, "logs/log.txt")
logging.basicConfig(
    level = logging.INFO,
    format= '[%(asctime)s] %(pathname)s:%(lineno)d %(levelname)s %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        SafeFileHandler(logpath),  # 将日志输出到文件
        logging.StreamHandler()  # 将日志输出到控制台
    ]
)

logger = logging.getLogger(__name__)
handler = RotatingFileHandler(logpath, maxBytes=1024*1024, backupCount=1)
logger.addHandler(handler)