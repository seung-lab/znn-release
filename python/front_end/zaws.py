import os

def s3download(s3fname, tmpdir="/tmp/"):
    """
    download aws s3 file

    params:
    - fname: string, file name in s3
    - tmpdir: string, temporary directory

    return:
    - lcfname: string, local file name
    """
    if "s3://" in s3fname:
        # base name
        lcfname = os.path.join( tmpdir, os.path.basename(s3fname) )
        os.system("aws s3 cp {} {}".format(s3fname, lcfname))
        return lcfname
    else:
        # this is not a s3 file, just a local file
        return s3fname