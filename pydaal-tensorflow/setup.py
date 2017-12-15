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

from setuptools import setup, find_packages
from os import environ as env

dependencies = ['tensorflow']

if 'PYDAAL_TEST' not in env:
  dependencies.append('pydaalcontrib')

setup(name='pydaaltensorflow',
      version='2017.0',
      description='Converter from Tensorflow to Intel DAAL',
      url='http://github.com/01org/pydaal-tensorflow',
      author='Vilen Jumutc',
      author_email='vilen.jumutcs@intel.com',
      classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
      ],
      install_requires=dependencies,
      test_suite='pydaaltensorflow',
      license='Apache-2.0',
      packages=find_packages(),
      zip_safe=True)