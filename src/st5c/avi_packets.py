"""RIFF AVI packet demultiplexing with no image codec or pixel decoder."""
import struct
from pathlib import Path


def jpeg_images(data):
    """Count top-level JPEG codestreams, skipping marker payloads and entropy."""
    count=0;pos=0;length=len(data)
    while pos<length:
        start=data.find(b'\xff\xd8',pos)
        if start<0:break
        pos=start+2;closed=False
        while pos<length:
            marker=data.find(b'\xff',pos)
            if marker<0:break
            pos=marker+1
            while pos<length and data[pos]==255:pos+=1
            if pos>=length:break
            tag=data[pos];pos+=1
            if tag==0 or 0xd0<=tag<=0xd7:continue
            if tag==0xd9:
                count+=1;closed=True;break
            if tag==0xd8:return count+1,False
            if tag==1:continue
            if pos+2>length:break
            segment=int.from_bytes(data[pos:pos+2],'big')
            if segment<2 or pos+segment>length:break
            pos+=segment
        if not closed:return count,False
    return count,count>0


def packet_metadata(path):
    data=Path(path).read_bytes();streams=[];packets=[];index=[];warnings=[];streaming_lengths=[]
    def chunks(begin,end):
        position=begin
        while position+8<=end:
            tag=data[position:position+4];size=struct.unpack_from('<I',data,position+4)[0]
            stop=position+8+size
            if size==0xffffffff and tag in (b'RIFF',b'LIST'):
                streaming_lengths.append(tag.decode('ascii'));stop=end
            if stop>end:
                if tag[2:] in (b'dc',b'db') and end>position+8:
                    warnings.append('truncated_final_packet_payload')
                    yield tag,position+8,end,end-position-8
                else:warnings.append('chunk_exceeds_container')
                return
            yield tag,position+8,stop,size
            position=stop+(size&1)
    def walk(begin,end,in_movi=False):
        for tag,start,stop,size in chunks(begin,end):
            if tag in (b'RIFF',b'LIST'):
                kind=data[start:start+4]
                if kind==b'strl':
                    values={t:data[a:b] for t,a,b,n in chunks(start+4,stop)}
                    sh=values.get(b'strh',b'');sf=values.get(b'strf',b'')
                    stream=dict(kind=sh[:4],codec=sh[4:8],scale=0,rate=0,start=0)
                    if len(sh)>=36:stream.update(scale=struct.unpack_from('<I',sh,20)[0],rate=struct.unpack_from('<I',sh,24)[0],start=struct.unpack_from('<I',sh,28)[0])
                    if sh[:4]==b'vids' and len(sf)>=20:stream['codec']=sf[16:20]
                    streams.append(stream)
                elif kind in (b'AVI ',b'AVIX',b'hdrl',b'movi',b'rec '):walk(start+4,stop,in_movi or kind in (b'movi',b'rec '))
            elif tag==b'idx1':
                if size%16:warnings.append('partial_legacy_index_entry')
                for at in range(start,stop-15,16):
                    ckid,flags,offset,n=struct.unpack_from('<4sIII',data,at);index.append((ckid,flags,n))
            elif in_movi and tag[2:] in (b'dc',b'db'):
                try:sid=int(tag[:2],16)
                except ValueError:warnings.append('unparsed_stream_number');continue
                packets.append((sid,start,stop,tag))
    if data[:4]!=b'RIFF' or data[8:12]!=b'AVI ':raise ValueError('Unsupported non-AVI container')
    walk(0,len(data))
    video_ids={i for i,r in enumerate(streams) if r['kind']==b'vids'}
    if len(video_ids)!=1:raise ValueError('Expected exactly one AVI video stream')
    sid=next(iter(video_ids));stream=streams[sid];video=[r for r in packets if r[0]==sid]
    positive=[r for r in video if r[2]>r[1]];zero=len(video)-len(positive)
    codec=stream['codec'].decode('ascii',errors='replace');bundled=0;envelope_unknown=0;codec_keyframes=0;vop_counts=[]
    if codec.upper() in ('MJPG','JPEG'):
        for _,a,b,_ in positive:
            count,ok=jpeg_images(data[a:b]);bundled+=int(count>1);envelope_unknown+=int(not ok or count!=1)
    elif codec.upper() in ('FMP4','DIVX','XVID','MP4V'):
        for _,a,b,_ in positive:
            packet=data[a:b];count=packet.count(b'\x00\x00\x01\xb6');vop_counts.append(count)
            bundled+=int(count>1);envelope_unknown+=int(count!=1)
            location=packet.find(b'\x00\x00\x01\xb6')
            if location>=0 and location+4<len(packet):codec_keyframes+=int((packet[location+4]>>6)==0)
    else:envelope_unknown=len(positive)
    prefix=f'{sid:02x}'.encode();ix=[r for r in index if r[0][:2]==prefix and r[0][2:] in (b'dc',b'db')]
    if len(ix)==len(video):keyframes=sum(bool(flags&16) for _,flags,size in ix if size>0);key_basis='AVI_idx1_flags'
    elif codec.upper() in ('MJPG','JPEG') and envelope_unknown==0:keyframes=len(positive);key_basis='verified_JPEG_intra_codestreams'
    elif vop_counts and envelope_unknown==0:keyframes=codec_keyframes;key_basis='MPEG4_VOP_coding_type_bits'
    else:keyframes=None;key_basis='unavailable'
    nominal_time=stream['scale']>0 and stream['rate']>0
    return dict(packet_count=len(video),positive_packet_count=len(positive),zero_size_packet_count=zero,
        keyframe_count=keyframes,keyframe_basis=key_basis,codec_tag=codec,
        explicit_packet_PTS_present=False,demux_PTS_present=nominal_time,
        demux_PTS_basis='AVI_stream_start_scale_rate_packet_order' if nominal_time else 'unavailable',
        demux_PTS_strict_monotone=nominal_time and bool(len(video)),
        demux_first_PTS_seconds=stream['start']*stream['scale']/stream['rate'] if nominal_time else None,
        demux_last_PTS_seconds=(stream['start']+len(video)-1)*stream['scale']/stream['rate'] if nominal_time and video else None,
        bundled_frame_packet_count=bundled,codec_envelope_unconfirmed_packets=envelope_unknown,
        MPEG4_VOP_count=sum(vop_counts) if vop_counts else None,
        frame_count_ambiguous=bool(zero or bundled),
        container_warnings=';'.join(sorted(set(warnings))),streaming_length_placeholders=len(streaming_lengths),video_pixels_decoded=False)
