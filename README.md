# Dream Flow

A google deep dream video script.


optional arguments:
  -i INPUT, --input INPUT
                        Input directory where extracted frames are stored
  -o OUTPUT, --output OUTPUT
                        Output directory where processed frames are to be
                        stored
  -t MODEL_PATH, --model_path MODEL_PATH
                        Model directory to use
  -m MODEL_NAME, --model_name MODEL_NAME
                        Caffe Model name to use
  -oct OCTAVES, --octaves OCTAVES
                        Octaves. Default: 4
  -octs OCTAVESCALE, --octavescale OCTAVESCALE
                        Octave Scale. Default: 1.4
  -itr ITERATIONS, --iterations ITERATIONS
                        Iterations. Default: 10
  -j JITTER, --jitter JITTER
                        Jitter. Default: 32
  -s STEPSIZE, --stepsize STEPSIZE
                        Step Size. Default: 1.5
  -b BLEND, --blend BLEND
                        Blend Amount. Default: "0.5" (constant), or "loop"
                        (0.5-1.0), or "random"
  -l LAYERS , --layers LAYERS
                        Layer to use ie 'inception_4c/output'
  -v VERBOSE, --verbose VERBOSE
                        verbosity [0-3]
  -gi GUIDE_IMAGE, --guide_image GUIDE_IMAGE
                        path to guide image
  -sf START_FRAME, --start_frame START_FRAME
                        starting frame nr
  -ef END_FRAME, --end_frame END_FRAME
                        end frame nr
</pre>
