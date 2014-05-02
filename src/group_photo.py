import sys
from gui import *
from PyQt4 import QtGui


def main(argv):
    '''
    parser = argparse.ArgumentParser(
        description='This is a application for group photo.')
    parser.add_argument('-g', '--group', action='store', default='group.png',
                        help='Select the group photo')
    parser.add_argument(
        '-p', '--person', action='append', default=['yangdawei.jpg'],
        help='Select people to add to the photo')

    result = parser.parse_args(argv)
    '''

    # group: the group photo to edit
    # person: the people who are excluded in the group photo
    '''
    photoGui = PhotoGui(result.group, set(result.person))
    photoGui.begin()
    '''
    app = QtGui.QApplication(sys.argv)

    w = GroupPhotoWindow()
    w.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main(sys.argv[1:])
