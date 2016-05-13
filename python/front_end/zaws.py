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
        bn = os.path.basename(s3fname)
        # local directory
        lcdir = os.path.dirname( s3fname )
        lcdir = lcdir.replace("s3://", "")
        lcdir = os.path.join(tmpdir, lcdir)
        # make directory
        os.makedirs(lcdir)
        # local file name
        lcfname = os.path.join( tmpdir, bn )
        # copy file from s3
        os.system("aws s3 cp {} {}".format(s3fname, lcfname))
        return lcfname
    else:
        # this is not a s3 file, just a local file
        return s3fname