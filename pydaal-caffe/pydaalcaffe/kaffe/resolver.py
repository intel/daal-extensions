# Copyright 2014-2017 Intel Corporation All Rights Reserved.
#
# The source code,  information  and material  ("Material") contained  herein is
# owned by Intel Corporation or its  suppliers or licensors,  and  title to such
# Material remains with Intel  Corporation or its  suppliers or  licensors.  The
# Material  contains  proprietary  information  of  Intel or  its suppliers  and
# licensors.  The Material is protected by  worldwide copyright  laws and treaty
# provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
# modified, published,  uploaded, posted, transmitted,  distributed or disclosed
# in any way without Intel's prior express written permission.  No license under
# any patent,  copyright or other  intellectual property rights  in the Material
# is granted to  or  conferred  upon  you,  either   expressly,  by implication,
# inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
# property rights must be express and approved by Intel in writing.
#
# Unless otherwise agreed by Intel in writing,  you may not remove or alter this
# notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
# suppliers or licensors in any way.

import sys
from os import environ as env

class CaffeResolver(object):
    def __init__(self):
        self.import_caffe()

    def import_caffe(self):
        self.caffe = None
        try:
            # Try to import PyCaffe first
            import caffe
            self.caffe = caffe
        except ImportError:
            # Fall back to the protobuf implementation
            from . import caffe_pb2
            self.caffe_pb = caffe_pb2

            if 'PYDAAL_TEST' not in env:
                _show_fallback_msg()

        if self.caffe:
            self.caffe_pb = self.caffe.proto.caffe_pb2

        self.NetParameter = self.caffe_pb.NetParameter

    def has_pycaffe(self):
        return self.caffe is not None

def _show_fallback_msg():
    msg = '''
------------------------------------------------------------
    WARNING: PyCaffe wrapper is not found!
    Falling back to a pure protobuf implementation.
------------------------------------------------------------

    '''
    sys.stderr.write(msg)
