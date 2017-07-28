/* ----------------------------------------------------------------- */
/*             The Speech Signal Processing Toolkit (SPTK)           */
/*             developed by SPTK Working Group                       */
/*             http://sp-tk.sourceforge.net/                         */
/* ----------------------------------------------------------------- */
/*                                                                   */
/*  Copyright (c) 1984-2007  Tokyo Institute of Technology           */
/*                           Interdisciplinary Graduate School of    */
/*                           Science and Engineering                 */
/*                                                                   */
/*                1996-2016  Nagoya Institute of Technology          */
/*                           Department of Computer Science          */
/*                                                                   */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/* - Redistributions of source code must retain the above copyright  */
/*   notice, this list of conditions and the following disclaimer.   */
/* - Redistributions in binary form must reproduce the above         */
/*   copyright notice, this list of conditions and the following     */
/*   disclaimer in the documentation and/or other materials provided */
/*   with the distribution.                                          */
/* - Neither the name of the SPTK working group nor the names of its */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission.   */
/*                                                                   */
/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND            */
/* CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,       */
/* INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF          */
/* MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE          */
/* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS */
/* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,          */
/* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED   */
/* TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,     */
/* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON */
/* ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,   */
/* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY    */
/* OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           */
/* POSSIBILITY OF SUCH DAMAGE.                                       */
/* ----------------------------------------------------------------- */

/************************************************************************
*                                                                       *
*    Pitch Mark Extraction                                              *
*                                                                       *
*                                                                       *
*       usage:                                                          *
*               pitch_mark [ options ] [ infile ] > stdout              *
*       options:                                                        *
*               -s  s     :  sampling frequency (Hz)       [16]         *
*               -t  t     :  voiced/unvoiced threshold     [0.9]        *
*                            value of t should be -0.5 < t < 1.6        *
*               -L  L     :  minimum fundamental frequency [60]         *
*                            to search for (Hz)                         *
*               -H  H     :  maximum fundamental frequency [240]        *
*                            to search for (Hz)                         *
*               -o  o     :  output format                 [0]          *
*                              0 pulse sequence: if a current sample is *
*                                a pitch mark, 1 or -1 is outputted     *
*                                considering polarity, otherwise 0 is   *
*                                outputted.                             *
*                              1 second when a pitch mark appears       *
*                              2 sample number when a pitch mark        *
*                                appears                                *
*       infile:                                                         *
*               data sequence                                           *
*                       x(0), x(1), ..., x(n-1), ...                    *
*       stdout:                                                         *
*               pitch mark                                              *
*                                                                       *
************************************************************************/

static char *rcs_id =
    "$Id: pitch_mark.c,v 1.3 2016/12/25 05:00:19 uratec Exp $";


/*  Standard C Libraries  */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef HAVE_STRING_H
#include <string.h>
#else
#include <strings.h>
#ifndef HAVE_STRRCHR
#define strrchr rindex
#endif
#endif


#if defined(WIN32)
#include "SPTK.h"
#else
#include <SPTK.h>
#endif

/*  Default Values  */
#define LOW    60.0
#define HIGH   240.0
#define SAMPLE_FREQ 16.0
#define OTYPE 0
#define THRESHOLD 0.9

/*  Command Name  */
char *cmnd;

typedef struct _float_list {
   float f;
   struct _float_list *next;
} float_list;

void usage(int status)
{
   fprintf(stderr, "\n");
   fprintf(stderr, " %s - pitch mark extraction\n", cmnd);
   fprintf(stderr, "\n");
   fprintf(stderr, "  usage:\n");
   fprintf(stderr, "       %s [ options ] [ infile ] > stdout\n", cmnd);
   fprintf(stderr, "  options:\n");
   fprintf(stderr, "       -s s  : sampling frequency (kHz)        [%.1f]\n",
           SAMPLE_FREQ);
   fprintf(stderr, "       -t t  : voiced/unvoiced threshold       [%.1f]\n",
           THRESHOLD);
   fprintf(stderr, "               value of t should be -0.5 < t < 1.6\n");
   fprintf(stderr, "       -L L  : minimum fundamental             [%g]\n",
           LOW);
   fprintf(stderr, "               frequency to search for (Hz)\n");
   fprintf(stderr, "       -H H  : maximum fundamental             [%g]\n",
           HIGH);
   fprintf(stderr, "               frequency to search for (Hz)\n");
   fprintf(stderr, "       -o o  : output format                   [%d]\n",
           OTYPE);
   fprintf(stderr,
           "                 0 pulse sequence: if a current sample is a pitch mark,\n");
   fprintf(stderr,
           "                   1 or -1 is outputted considering polarity,\n");
   fprintf(stderr, "                   otherwise 0 is outputted.\n");
   fprintf(stderr, "                 1 second when a pitch mark appears\n");
   fprintf(stderr,
           "                 2 sample number when a pitch mark appears\n");
   fprintf(stderr, "       -h    : print this message\n");
   fprintf(stderr, "  infile:\n");
   fprintf(stderr, "       waveform (%s)             \n", FORMAT);
   fprintf(stderr, "  stdout:\n");
   fprintf(stderr, "       pitch mark (%s)\n", FORMAT);
   fprintf(stderr, "  notice:\n");
   fprintf(stderr,
           "       Regarding -t option, when the threshold is raised,\n");
   fprintf(stderr, "         the number of pitch marks increases.\n");
#ifdef PACKAGE_VERSION
   fprintf(stderr, "\n");
   fprintf(stderr, " SPTK: version %s\n", PACKAGE_VERSION);
   fprintf(stderr, " CVS Info: %s", rcs_id);
#endif
   fprintf(stderr, "\n");
   exit(status);
}


int main(int argc, char **argv)
{
   int length, otype = OTYPE;
   double *x, threshold = THRESHOLD, sample_freq = SAMPLE_FREQ,
       L = LOW, H = HIGH;
   FILE *fp = stdin;
   float_list *top, *cur, *prev;
   void pitch_mark(float_list * input, int length, double sample_freq,
                   double min, double max, double threshold, int otype);

   if ((cmnd = strrchr(argv[0], '/')) == NULL)
      cmnd = argv[0];
   else
      cmnd++;
   while (--argc)
      if (**++argv == '-') {
         switch (*(*argv + 1)) {
         case 's':
            sample_freq = atof(*++argv);
            --argc;
            break;
         case 't':
            threshold = atof(*++argv);
            --argc;
            break;
         case 'L':
            L = atof(*++argv);
            --argc;
            break;
         case 'H':
            H = atof(*++argv);
            --argc;
            break;
         case 'o':
            otype = atoi(*++argv);
            --argc;
            break;
         case 'h':
            usage(0);
         default:
            fprintf(stderr, "%s : Invalid option '%c'!\n", cmnd, *(*argv + 1));
            usage(1);
         }
      } else {
         fp = getfp(*argv, "rb");
      }
   sample_freq *= 1000.0;

   x = dgetmem(1);
   top = prev = (float_list *) malloc(sizeof(float_list));
   length = 0;
   prev->next = NULL;
   while (freadf(x, sizeof(*x), 1, fp) == 1) {
      cur = (float_list *) malloc(sizeof(float_list));
      cur->f = (float) x[0];
      length++;
      prev->next = cur;
      cur->next = NULL;
      prev = cur;
   }

   pitch_mark(top->next, length, sample_freq, L, H, threshold, otype);

   return (0);
}
