AM_CPPFLAGS = -I @top_srcdir@/include
bin_PROGRAMS = pitch_mark
pitch_mark_SOURCES = pitch_mark.c reaper/epoch_tracker_main.cc reaper/core/file_resource.cc \
	reaper/core/file_resource.h reaper/core/float_matrix.cc \
	reaper/core/float_matrix.h reaper/core/float_matrix-inl.h \
	reaper/core/track.cc reaper/core/track.h \
	reaper/epoch_tracker/epoch_tracker.cc reaper/epoch_tracker/epoch_tracker.h \
	reaper/epoch_tracker/fd_filter.cc reaper/epoch_tracker/fd_filter.h \
	reaper/epoch_tracker/fft.cc reaper/epoch_tracker/fft.h \
	reaper/epoch_tracker/lpc_analyzer.cc reaper/epoch_tracker/lpc_analyzer.h \
	reaper/wave/codec_api.h reaper/wave/codec_api-inl.h \
	reaper/wave/codec_riff.cc reaper/wave/codec_riff.h \
	reaper/wave/wave.cc reaper/wave/wave.h reaper/wave/wave_io.cc \
	reaper/wave/wave_io.h reaper/wave/wave_io-inl.h
pitch_mark_LDADD = @top_builddir@/lib/libSPTK.a
