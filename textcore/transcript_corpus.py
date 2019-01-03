import cottoncandy as cc

def get_transcript_uris(bucket='stimulidb', extension='txt'):
    cci = cc.get_interface(bucket)

    # get all the objects in `bucket` that end with `extension`
    all_names = cci.glob('*.%s' % extension)

    # fix each name into a URI and return the list
    fixed_names = [cc.utils.pathjoin('s3://', bucket, n) for n in all_names]

    return fixed_names