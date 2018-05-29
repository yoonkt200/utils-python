#from pyhive import hive
import sys ; sys.path.append('../')
import config


if __name__ == '__main__':
    test = config.RECMNDTN_MNTRNG_CONFIG['volm']['asscr']
    print(test)