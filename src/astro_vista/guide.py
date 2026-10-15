"""Short documentation, shown by the ? buttons and the Help menu."""

from __future__ import annotations

import re

from PySide6 import QtCore, QtWidgets

from . import reference

TOPICS = {
    "overview": (
        "AstroVista Lite",
        """
<p>The lite edition of AstroVista, an astronomical image analysis studio in
development. It fits elliptical <b>isophotes</b> to galaxies (ELLIPROF) and <b>finds
sources</b> around them, with layers, masks and a terminal.</p>
<p><b>Installing:</b> AstroVista Lite is on PyPI for Linux and macOS:
<code>pip install AstroVista</code>, then <code>astrovista</code> (or
<code>astrovista --example</code>). Everything comes with it; nothing else to install.</p>
<p><b>A first session:</b> File ▸ Open example, then <i>Fit isophotes</i>. Try the
Source finder, pick the centre, <i>Mask the found sources</i>, and fit again.</p>
<p><b>The window:</b> buffers and layers on the left, the image in the middle, and on the
right four tabs: <i>Inspector</i>, <i>Tools</i> (ELLIPROF, Source finder, Display, Contours),
<i>Results</i> (the profile, the sources, plots) and <i>Header</i>; the terminal at the
bottom. Panels are resized by dragging
their edges, folded with the arrow in their title bar (or a double-click on it), and
popped out as windows of their own with ⧉ (for another monitor; ⤓ or a double-click puts
them back). Each panel docks only in its own place; View ▸ Reset the layout restores
the standard layout. The buttons at the right of the status bar fold and unfold them. The command
line is the last line of the Terminal, as in a shell (it goes with it when popped out);
when the results are folded, it waits in the narrow bar at the bottom of the window.</p>
<p><b>Coordinates:</b> X0, Y0 and the mask commands use ELLIPROF's coordinates, in which
the first pixel spans 0 to 1. The status bar also shows the pixel's column and row
(from 1), its value and its sky position.</p>""",
    ),
    "lesson": (
        "A first lesson: the light of a galaxy",
        """
<p>About thirty minutes, for a class or on your own, with the example image of the galaxy
UGC 12517 (Hubble Space Telescope, near infrared).</p>
<ol>
<li><b>Look at the image.</b> File ▸ Open example. Move the mouse: the status bar shows
the pixel value and the sky position. Change <i>Scale</i> from Linear to Log or Asinh:
faint light around the galaxy appears. Drag on the image with the right mouse button to
change the contrast. <i>Question:</i> how far out does the galaxy's light go?</li>
<li><b>Find the sources.</b> Open the Source finder tab and press <i>Find sources</i>.
Each ellipse is a star or a galaxy. Click one to see its row in the Sources table. In
the Plots tab, plot <code>MAG_AUTO</code> against <code>FLUX_RADIUS</code>: stars form
a narrow sequence. <i>Question:</i> which sources are stars?</li>
<li><b>Find the centre.</b> In the ELLIPROF tab press <i>Pick on the image</i> and click
the galaxy's bright core: with <i>Snap to sources</i> the small red cross jumps to the
centre the Source finder measured. Pick R1, the largest ellipse, by clicking where the galaxy
fades into the sky.</li>
<li><b>Mask what is not the galaxy.</b> Press <i>Mask the found sources</i> (every
other source becomes a mask layer). Touch it up with the brush and the eraser. The
masks show as coloured layers; untick one to hide it.</li>
<li><b>Fit the isophotes.</b> Press <i>Fit isophotes</i>. Green ellipses follow the
galaxy's light. The Profile tab plots the intensity against the radius; tick
<i>R<sup>¼</sup> axis</i>: an elliptical galaxy's profile is close to a straight line
(de Vaucouleurs' law). Plot the ellipticity and the position angle too.</li>
<li><b>Look at what is left.</b> Choose the residual buffer (image minus model) on the
left. Press ← and → (after clicking the image) to blink between the image, the model and
the residual. <i>Question:</i> what does the model miss: dust, a disc, companions?</li>
</ol>
<p>Every step was a command: the Terminal tab lists them, and File ▸ Save the commands
keeps the lesson as a file that runs again with File ▸ Run a command file.</p>""",
    ),
    "images": (
        "Images, buffers and coordinates",
        """
<p><b>Opening images:</b> File ▸ Open (or drop files on the window). Any FITS file opens,
whatever its name, also gzip-compressed. A file with several images asks which one;
<code>rd 1 file.fits[sci]</code> or <code>[2]</code> choose one on the command line.
Legacy <code>.dmask</code> bitmaps open as mask layers.</p>
<p><b>Buffers:</b> every image lives in a numbered buffer, listed on the left with a small
picture. Results (a model, a residual, arithmetic) go to new buffers, so nothing is
overwritten unless you ask. Right-click a buffer to save it, see its statistics, use it as
a mask, copy or close it.</p>
<p><b>Coordinates:</b> x and y run along the image's columns and rows, with the first
pixel spanning 0 to 1 (its centre at 0.5). FITS and DS9 count from 1 instead: the status
bar shows both, and the pixel's value and sky position (RA, Dec) when the image has a
WCS.</p>
<p><b>Saving:</b> File ▸ Save buffer writes a FITS file; the names suggested follow the
image (u12517j_residual.fits). The profile, the isophotes (DS9 regions) and the sources
(CSV) are under File ▸ Export.</p>""",
    ),
    "arithmetic": (
        "Image arithmetic",
        """
<p>Image arithmetic does sums with whole images: type a formula in the terminal and it is
worked out <b>pixel by pixel</b>, every pixel on its own, and the result goes into the
buffer on the left of the <code>=</code>.</p>
<pre>[3] = [1] - [2]</pre>
<p>reads "buffer 3 becomes buffer 1 minus buffer 2": each pixel of buffer 3 is the pixel
of buffer 1 at the same place minus the pixel of buffer 2 there. If buffer 3 is empty a
new buffer is made; if it holds an image, that image is replaced (so
<code>[1] = [1] * 2</code> doubles buffer 1 in place). <code>B3 = B1 - B2</code> is the
same formula: <code>Bn</code> and <code>[n]</code> both mean buffer n. <code>calc 3
EXPR='[1] - [2]'</code> is the same again, for command files.</p>

<h4>What a formula can hold</h4>
<table cellspacing="4">
<tr><td><code>[1]</code> or <code>B1</code></td><td>a buffer: its pixel at each place</td></tr>
<tr><td><code>3246</code>, <code>2.5</code>, <code>1e-3</code></td><td>numbers, the same for
every pixel</td></tr>
<tr><td><code>k</code></td><td>a variable, set before with <code>set k=2.5</code>
(<code>typ k</code> shows it)</td></tr>
<tr><td><code>pi</code>, <code>e</code>, <code>nan</code></td><td>constants</td></tr>
<tr><td><code>mymask</code></td><td>a mask layer, by its name: 1 where it is good, NaN where
it masks (see below)</td></tr>
<tr><td><code>sqrt([1])</code></td><td>functions (listed below)</td></tr>
</table>

<h4>Operators, and which comes first</h4>
<table cellspacing="4">
<tr><td><code>( )</code></td><td>parentheses: worked out first</td></tr>
<tr><td><code>^</code> or <code>**</code></td><td>powers: <code>[1]^2</code> squares
each pixel</td></tr>
<tr><td><code>-x</code></td><td>minus sign (after powers: <code>-[1]^2</code> is
−([1]²))</td></tr>
<tr><td><code>* / %</code></td><td>multiply, divide, remainder</td></tr>
<tr><td><code>+ -</code></td><td>add, subtract</td></tr>
<tr><td><code>&lt; &lt;= &gt; &gt;= == !=</code></td><td>comparisons: 1 where true, 0
where false (last)</td></tr>
</table>
<p>As on paper: <code>[1] - 100 * 2</code> subtracts 200; <code>([1] - 100) * 2</code>
subtracts 100 first, then doubles. When unsure, add parentheses.</p>

<h4>Conditions</h4>
<p>A comparison gives an image of 1 and 0, which can pick pixels:</p>
<pre>[5] = [1] &gt; 1000                    # 1 where brighter than 1000, else 0
[6] = where([1] &gt; 1000, [1], 0)     # keep the bright pixels, 0 elsewhere
[7] = where([1] &gt; 1000, nan, [1])   # hide the bright pixels (NaN)
[8] = ([1] &gt; 10) * ([1] &lt; 100)      # both true: multiply the conditions ("and")
[9] = clip(([1] &lt; 10) + ([1] &gt; 100), 0, 1)   # either true: add and clip ("or")</pre>
<p><code>where(condition, a, b)</code> takes <code>a</code> where the condition is true
(non-zero) and <code>b</code> elsewhere. There is no <code>and</code> or
<code>or</code>: multiply or add the conditions as above. A comparison with a NaN pixel is
false (0).</p>

<h4>Functions, pixel by pixel</h4>
<p><code>abs sqrt exp log</code> (natural) <code>log10 sin cos tan arcsin arccos arctan
sinh cosh tanh floor ceil round isnan isfinite nan_to_num</code> act on each pixel;
<code>minimum(a, b)</code> and <code>maximum(a, b)</code> take the smaller or larger of two
values at each pixel; <code>clip(a, low, high)</code> limits values to a range;
<code>div(a, b, fill)</code> divides, with <code>fill</code> where b is 0.</p>

<h4>mean, median, sum, min, max, std: two ways</h4>
<p>With <b>several</b> values they work pixel by pixel, across the images:
<code>median([1], [2], [3])</code> is at each pixel the median of the three images there
(to combine exposures; a number counts as an image of that value). With <b>one</b>
buffer they give a single number over the whole image: <code>median([1])</code> is the
median of all of buffer 1's pixels, so <code>[2] = [1] - median([1])</code> subtracts it
from every pixel. <code>avg</code> is <code>mean</code>.</p>

<h4>Masks in formulas</h4>
<p>A mask layer's name stands for 1 where it is good and NaN where it masks, so
multiplying hides the masked pixels: <code>[4] = [1] * mymask</code>. For 0 and 1
instead: <code>good(mymask)</code> is 1 where good, <code>masked(mymask)</code> 1 where it
masks; <code>[4] = where(masked(mymask), 0, [1])</code> puts 0 on the masked pixels. In
AstroVista Lite masks are always 0 for bad and 1 for good.</p>

<h4>Special values: NaN and division by zero</h4>
<table cellspacing="4">
<tr><td><b>What</b></td><td><b>Gives</b></td></tr>
<tr><td>x / 0, 0 / 0, log(0), sqrt(−1), an overflow</td><td>NaN, never infinity</td></tr>
<tr><td>+ − × ÷ with a NaN pixel</td><td>NaN</td></tr>
<tr><td>mean, median, sum, min, max, std</td><td>NaN pixels left out; NaN only where every
value is NaN</td></tr>
<tr><td>a comparison with NaN</td><td>0 (false)</td></tr>
<tr><td>a mask layer as a value</td><td>1 where good, NaN where it masks</td></tr>
</table>
<p>NaN ("not a number") marks a pixel without a value. It is shown as the darkest colour,
left out of statistics (<code>stats</code>, <code>sky</code>), and masked in ELLIPROF.
<code>nan_to_num([1])</code> turns NaN into 0; <code>isnan([1])</code> shows where they
are.</p>

<h4>Sizes, headers and names</h4>
<p>Every buffer in a formula must have the same size. The result takes the header (and
the sky coordinates) of the first buffer in the formula, and remembers the formula, shown
under its name in the buffer list and written as HISTORY when saved. Add
<code>tag="a name"</code> after the formula to name it (see <i>Tags</i>). A formula with
no buffer in it (<code>[3] = 5</code>) fills an existing buffer with that value.</p>

<h4>Worked examples</h4>
<pre>[2] = [1] - median([1]) tag="no sky"     # subtract a constant sky
[3] = [1] / [5]                          # image / model (the model in buffer 5)
[4] = -2.5 * log10([1] / [2])            # a colour: magnitudes of two filters
[5] = [1] / sqrt(abs([1]) + 25)          # a rough signal-to-noise map
[6] = median([1], [2], [3]) tag=stack    # combine three exposures, NaN left out
[7] = ([1] - [8]) / [8] * 100            # the difference in per cent
set gain=2.4
[9] = [1] * gain tag=electrons           # counts to electrons</pre>

<h4>When it goes wrong</h4>
<p><i>buffer 9 is empty</i>: the formula uses a buffer with nothing in it.
<i>the buffers in the expression differ in size</i>: all must be the same size.
<i>'x' is not a buffer, a variable or a mask layer</i>: a misspelt name, or
<code>set</code> it first. <i>cannot read the expression</i>: a parenthesis is missing,
or <code>tag=</code> is not separated by a space. Only the operators and functions on this
page can be used: a formula is read, never run as a program.</p>""",
    ),
    "tags": (
        "Tags: naming what you make",
        """
<p>Every buffer made by a command remembers how it was made: the line is shown in small
text under its name in the buffer list, as a tooltip in the Blink menu, and as a HISTORY
card in its FITS header when saved.</p>
<p>To give it a name of your own, add <code>tag="..."</code> to the line:</p>
<pre>[3] = [1] - median([1]) tag="sky-subtracted"
sources 1 TOSEG=4 tag="segmentation"
elliprof 1 X0=567 Y0=562 R0=9 R1=347 NR=23 TOMODEL=5 TORESID=6 tag=fit</pre>
<p>The name then appears everywhere instead of the buffer's file name. To rename a buffer
afterwards: <code>tag 3 "my name"</code>, or right-click it ▸ Tag…</p>""",
    ),
    "files": (
        "Opening files and extensions",
        """
<p><b>File ▸ Open</b> (Ctrl+O), a drop on the window, <b>File ▸ Open recent</b> (the last
ten), or the terminal: <code>rd 1 image.fits</code>. FITS files are recognised by what
they hold, whatever they are called, also when gzip-compressed.</p>
<p><b>Files with several images</b> (HST, JWST...): the window asks which one; in the
terminal choose it by number or by name:</p>
<pre>fitsinfo frame.fits          # list what the file holds
rd 1 frame.fits[1]           # by number
rd 1 frame.fits[sci]         # by name (EXTNAME)
rd 1 frame.fits[sci,2]       # by name and version (EXTVER)</pre>
<p><b>Starting with files loaded.</b> On the command line that starts the program,
<code>-i</code> loads images into the buffers (1, 2, ... in the order given) and
<code>-m</code> loads masks (FITS or <code>.dmask</code>, 0 bad, 1 good) as mask layers,
also shown as buffers and ticked for the fit:</p>
<pre>astrovista -i galaxy.fits "frame.fits[sci]" -m galaxy.dmask
astrovista -i g.fits -i r.fits -m edges.fits     # -i and -m can be repeated
astrovista galaxy.fits                            # images without -i work too</pre>
<p><code>--example</code> loads the example, <code>--run file</code> runs a command file,
and <code>astrovista --help</code> lists it all.</p>""",
    ),
    "listing": (
        "Files in the terminal",
        """
<p><code>ls</code>, <code>ll</code> (ls -lrt) and <code>la</code> list the current folder
in colour: <span style='color:#61afef'><b>folders</b></span>,
<span style='color:#98c379'>FITS files</span>, <span style='color:#c678dd'>pictures</span>,
<span style='color:#e5c07b'>text and tables</span>, <span style='color:#e06c75'>programs</span>.
Click a folder to go into it (and list it); click a FITS file to open it.</p>
<p><code>cd folder</code> changes the folder file names are relative to, <code>pwd</code>
shows it. Any other program on the PATH runs too (<code>grep</code>, <code>head</code>...);
<code>!</code> forces the shell; Ctrl+C stops a program.</p>""",
    ),
    "blinking": (
        "Blinking images",
        """
<p>Blinking flips between images in the same place, so that differences jump out (a
residual against the image, two filters, two epochs).</p>
<ol>
<li>Tick <b>Blink</b> above the image.</li>
<li>In the menu beside it, choose the buffers (all of them when none is chosen). Keep
<b>Same scale for every buffer</b> on, so equal brightness looks equal.</li>
<li>Click the image, then press ← and →.</li>
</ol>
<p><b>Zoomed in?</b> Blinking keeps the place and the zoom: when both images have sky
coordinates (WCS) that match, it locks on the sky, so the same stars stay in the same
place even if the images differ in size, pixel scale or pointing; otherwise it locks on
the image, the same pixel at the same zoom. The status bar says which. When the whole
image is in view, each image is shown whole.</p>""",
    ),
    "recipes": (
        "Recipes",
        """
<p><b>Measure a galaxy's profile.</b> Open the image ▸ Source finder ▸ <i>Find
sources</i> ▸ ELLIPROF: <i>Pick on the image</i> (snaps to the galaxy), <i>Pick</i> R1 where
it fades ▸ <i>Estimate</i> the sky ▸ <i>Mask the found sources</i> ▸ <i>Fit isophotes</i> ▸
Results ▸ Profile, tick <i>R¼ law</i>.</p>
<p><b>See what the model misses.</b> After a fit, tick Blink, choose the image and the
residual, and press ← →. Or <code>[9] = [1] / [5] tag="ratio"</code>, with the model in buffer 5.</p>
<p><b>Mask a bright source by hand.</b> Choose the circle tool (M), set its radius, click the
source; Alt-click to unmask; Ctrl+Z undoes.</p>
<p><b>Find only bright, compact sources.</b> Source finder: raise the detection threshold; or Masks ▸
Mask the sources where… <code>MAG_AUTO &lt; 18 and FWHM_IMAGE &lt; 3</code>.</p>
<p><b>Subtract the sky and save.</b> <code>[2] = [1] - median([1]) tag="no sky"</code>,
then right-click buffer 2 ▸ Save as FITS.</p>
<p><b>Replay a session.</b> File ▸ Save the commands; later File ▸ Run a command file.</p>""",
    ),
    "imagecommands": (
        "Image commands",
        """
<p>Commands in the style of MONSTA (John Tonry's union of Vista and Mongo), written for
AstroVista Lite. N is a buffer number; <code>help name</code> explains each one, and the
<i>Command reference</i> gives every option with an example.</p>
<table cellspacing="4">
<tr><td><code>ac sc mc dc N value</code></td><td>add, subtract, multiply, divide by a
number</td></tr>
<tr><td><code>ai si mi di N M</code></td><td>the same with another buffer</td></tr>
<tr><td><code>log ln sqrt inverse round N</code></td><td>functions, in place</td></tr>
<tr><td><code>mn max min abx N</code></td><td>statistics (they also set variables)</td></tr>
<tr><td><code>print N C R</code></td><td>the pixel values around a column and row</td></tr>
<tr><td><code>shift N DX= DY=</code></td><td>move by (fractions of) pixels</td></tr>
<tr><td><code>rotate N ANGLE=</code>, <code>flip N X</code></td><td>turn or mirror</td></tr>
<tr><td><code>bin N 2</code>, <code>mash N ROWS</code></td><td>rebin; add up rows or
columns</td></tr>
<tr><td><code>smooth N FWHM=</code>, <code>filt N MEDIAN=</code></td><td>Gaussian or
median smoothing</td></tr>
<tr><td><code>zap N SIG=</code>, <code>clip N MAX=</code></td><td>remove hot pixels; clip
values</td></tr>
<tr><td><code>skysub N</code></td><td>subtract the sky</td></tr>
<tr><td><code>fft N OUT=M</code></td><td>the power spectrum</td></tr>
<tr><td><code>noise N SIGMA=</code>, <code>gauss N X= Y= FWHM= PEAK=</code></td><td>add
noise or a star (to practise)</td></tr>
<tr><td><code>ch N KEY=value</code>, <code>chead N</code></td><td>the FITS header</td></tr>
<tr><td><code>set x=1</code>, <code>typ</code></td><td>variables</td></tr>
</table>""",
    ),
    "profile": (
        "Reading the ELLIPROF profile",
        """
<p>Each row of the Profile table is one fitted ellipse (isophote):</p>
<table cellspacing="4">
<tr><td><b>Rmaj</b></td><td>its semi-major axis, pixels</td></tr>
<tr><td><b>I0</b></td><td>the brightness along it (with the sky subtracted)</td></tr>
<tr><td><b>ellip</b></td><td>its ellipticity, 1 − b/a: 0 is round</td></tr>
<tr><td><b>alpha</b></td><td>its orientation, degrees (the long axis lies at alpha + 90°
from the x axis)</td></tr>
<tr><td><b>x0, y0</b></td><td>its centre</td></tr>
<tr><td><b>I3, A3; I4, A4</b></td><td>the 3θ and 4θ harmonics: I4 shows whether the
isophote is discy or boxy</td></tr>
<tr><td><b>slope</b></td><td>how fast the brightness falls with radius</td></tr>
</table>
<p>Plot I0 against Rmaj<sup>¼</sup> (tick <i>R¼ axis</i>): an elliptical galaxy is close to
a straight line (de Vaucouleurs' law); a disc galaxy against Rmaj on a log scale. Tick
<b>R¼ law</b> to over-plot the de Vaucouleurs law fitted to the profile,
I(R) = Ie exp(−7.669 [(R/Re)<sup>¼</sup> − 1]): the note gives the effective radius Re (half
the light inside it), the intensity there Ie, and the scatter of the points about the
law in magnitudes (small means the law describes the galaxy well). A
surface brightness in magnitudes per square arcsecond is
μ = zero point − 2.5 log₁₀(I0 / pixel area).</p>""",
    ),
    "saving": (
        "Saving and exporting",
        """
<ul>
<li><b>An image</b>: File ▸ Save buffer (or right-click it ▸ Save as FITS); the FITS
header keeps how it was made.</li>
<li><b>A mask</b>: right-click it in Layers ▸ Save as… (.fits, or .dmask for the legacy
bitmap); 0 is bad, 1 good.</li>
<li><b>The profile</b>: in Results ▸ Profile the save button saves what is shown: with
<i>Table</i> ticked, <i>Save CSV…</i> writes the table as CSV with its columns aligned;
otherwise <i>Save PNG…</i> writes the plot. File ▸ Export the profile (CSV) writes the
table too, and the isophotes can be exported as DS9 regions.</li>
<li><b>The sources</b>: <i>Save CSV…</i> in Results ▸ Sources, or File ▸ Export the
sources (CSV): every column of the run, comma separated, with the columns aligned so the
file reads well in a text editor. Spreadsheets read it as it is; in Python use
<code>pandas.read_csv(file, skipinitialspace=True)</code> or <code>astropy.table.Table.read(file,
format="ascii.csv")</code>.</li>
<li><b>The plots</b>: <i>Save PNG…</i> in Results ▸ Profile and Results ▸ Plots (or File ▸
Export the profile plot / the sources plot) writes the plot as it is shown, as a PNG figure
drawn by matplotlib: the quantity and axis chosen, the R¼ law when ticked, the selected
source circled.</li>
<li><b>The session</b>: File ▸ Save the commands writes every command run; File ▸ Run a
command file (or <code>run file</code>) replays it.</li>
</ul>
<p>The window's layout is remembered between sessions; View ▸ Reset the layout restores
the standard one.</p>""",
    ),
    "inspect": (
        "Inspector, plots and header",
        """
<p><b>Inspector</b> (the first tab of the right-hand panel): the pixel under the mouse (or the crosshair),
its value and sky position, a magnified view in the display's colours, and the mean,
median and σ of the 5 × 5 pixels around it.</p>
<p><b>Results ▸ Profile</b>: the fitted profile as a plot (choose the quantity; tick
R¼ for de Vaucouleurs' law) or, ticking <i>Table</i>, its numbers. Click a row to see
that ellipse on the image.</p>
<p><b>Results ▸ Sources</b>: the sources of a run, sortable; click one to see it on the
image; double-click to take it as ELLIPROF's centre. <b>Results ▸ Plots</b>: any two
columns against each other (stars stand out in MAG_AUTO against FLUX_RADIUS); click a
point to find the source.</p>
<p><b>Header</b>: the FITS header of the buffer shown.</p>""",
    ),
    "glossary": (
        "Words explained",
        """
<dl>
<dt><b>Isophote</b></dt><dd>a line joining points of equal brightness.</dd>
<dt><b>Ellipticity</b></dt><dd>1 − b/a, from 0 (a circle) towards 1 (a thin line).</dd>
<dt><b>Position angle</b></dt><dd>the direction of an ellipse's long axis.</dd>
<dt><b>Sky (background)</b></dt><dd>the light everywhere in the image that is not from the
sources: the night sky, zodiacal light, the detector.</dd>
<dt><b>Mask</b></dt><dd>marks pixels to leave out. In AstroVista Lite 0 means bad
(masked) and 1 good, in files and buffers alike.</dd>
<dt><b>Magnitude</b></dt><dd>a logarithmic brightness: 5 magnitudes are a factor 100;
smaller is brighter. m = zero point − 2.5 log₁₀(flux).</dd>
<dt><b>Zero point</b></dt><dd>the magnitude of a source giving one count (per second).</dd>
<dt><b>FWHM</b></dt><dd>full width at half maximum: the width of a star's image.</dd>
<dt><b>Kron radius</b></dt><dd>a radius that holds most of a source's light, used by
MAG_AUTO.</dd>
<dt><b>Segmentation</b></dt><dd>an image giving, for every pixel, the number of the source
it belongs to.</dd>
<dt><b>WCS</b></dt><dd>World Coordinate System: the header's description of where each
pixel lies on the sky.</dd>
<dt><b>NaN</b></dt><dd>Not a Number: a pixel without a value (masked, or undefined).</dd>
<dt><b>FITS</b></dt><dd>the standard file format of astronomical images.</dd>
</dl>""",
    ),
    "keys": (
        "Mouse and keyboard",
        """
<table cellspacing="4">
<tr><td>Wheel</td><td>zoom the image (over a setting it scrolls the panel; settings never
change under the wheel)</td></tr>
<tr><td>Drag</td><td>pan (Pan or Crosshair tool)</td></tr>
<tr><td>Right-drag, or Ctrl/⌘/Control-drag</td><td>contrast and bias</td></tr>
<tr><td>Right-click</td><td>copy coordinates, set the centre, mask a circle...</td></tr>
<tr><td>Ctrl-click (⌘-click)</td><td>remove the masked patch under the cursor</td></tr>
<tr><td>Alt or Shift with a mask tool</td><td>unmask (mask, when the Unmask button is
on)</td></tr>
<tr><td>[ ] (and { })</td><td>turn a drawn ellipse, box or polygon by 5° (1°)</td></tr>
<tr><td>Enter, Esc</td><td>apply, cancel a drawn shape waiting to be applied (drag it to
move it, its squares to resize it)</td></tr>
<tr><td>← →</td><td>blink (when Blink is ticked)</td></tr>
<tr><td>Alt + arrows</td><td>nudge the crosshair</td></tr>
<tr><td>P C X M E B G R U</td><td>the tools: pan, pick, crosshair, circle, ellipse, box,
polygon, brush, eraser</td></tr>
<tr><td>Ctrl+Z, Ctrl+Shift+Z</td><td>undo, redo mask edits</td></tr>
<tr><td>Ctrl+E, Ctrl+D</td><td>fit isophotes, find sources</td></tr>
<tr><td>Ctrl+G</td><td>go to a pixel or a sky position</td></tr>
<tr><td>Ctrl+`</td><td>the terminal</td></tr>
<tr><td>Tab, ↑ ↓, Ctrl+C</td><td>in the terminal: complete, history, stop</td></tr>
<tr><td>F1</td><td>this documentation</td></tr>
</table>""",
    ),
    "problems": (
        "Troubleshooting",
        """
<p><b>The fit fails or gives strange ellipses.</b> Check the centre (X0, Y0) is on the
galaxy, R0 is not too small (start at 3–10 pixels), R1 does not go far into the sky, the
sky value is right (<i>Estimate</i>), and bright sources are masked.</p>
<p><b>The Source finder finds too much noise</b>: raise the detection threshold or the
minimum area. <b>It misses faint sources</b>: lower the threshold, keep the filter on.
<b>Big galaxies break into pieces</b>: raise the deblending contrast (0.01–0.1), or the
background mesh size.</p>
<p><b>The image looks black or white</b>: change Limits (99.5%, ZScale) and Scale (Log,
Asinh), or right-drag on the image; Reset puts contrast and bias back.</p>
<p><b>A mask does not seem to work</b>: it must be ticked in <i>Mask with</i> (ELLIPROF) or
named in <i>Leave out</i> (Source finder). Masks read 0 as bad and 1 as good.</p>
<p><b>A command is refused</b>: the Terminal says why; <code>help name</code> shows how to
write it, and Tab completes it.</p>
<p><b>A file does not open</b>: <code>fitsinfo file</code> shows what it holds; open one
image of a multi-extension file with <code>rd 1 file.fits[sci]</code>.</p>
<p><b>A panel has disappeared</b>: the Layers, Tools and Terminal buttons at the bottom
right open them again; View ▸ Reset the layout puts everything back.</p>
<p><b>The keys do nothing on the image</b>: click the image first, so that it has the
keyboard. The arrows blink only when Blink is ticked; Alt + arrows move the crosshair.</p>
<p><b>A setting changed when I scrolled</b>: it cannot: the wheel scrolls panels and zooms
the image, never changes a setting.</p>""",
    ),
    "credits": (
        "Credits",
        """
<p><b>AstroVista Lite</b> by Ehsan Kourkchi, on PyPI for Linux and macOS
(<code>pip install AstroVista</code>).</p>
<p><b>John Tonry.</b> ELLIPROF, the isophote fitting at the heart of AstroVista Lite, was
originally developed by John Tonry as part of MONSTA; it is used here through the elliprof
package, which compiles his original code unchanged. The terminal follows the command
language of MONSTA, John Tonry's image analysis and plotting system, which joined Vista
(originally by Richard Stover and Tod Lauer, with image display by John Tonry) and Mongo
(by John Tonry). The commands of AstroVista Lite are new code written in that style.</p>
<p><b>Citing AstroVista Lite:</b> please cite its GitHub repository:<br>
Kourkchi, E., <i>AstroVista Lite</i>, <a href="https://github.com/ekourkchi/AstroVista-Lite">github.com/ekourkchi/AstroVista-Lite</a><br>
and, for what you measure with it, the works below (elliprof for the isophotes, SEP and
SExtractor for the Source finder).</p>
<p><b>Built on:</b></p>
<ul>
<li><b>elliprof</b>: John Tonry's ELLIPROF as a Python package (<a
href="https://github.com/ekourkchi/elliprof">github.com/ekourkchi/elliprof</a>; the
package is MIT, the original ELLIPROF sources keep their own terms).</li>
<li><b>SEP</b>: source extraction and photometry, by K. Barbary and contributors
(<a href="https://github.com/sep-developers/sep">github.com/sep-developers/sep</a>,
LGPL-3.0); Barbary 2016, JOSS 1, 58. It implements the algorithms of
<b>SExtractor</b> (Bertin &amp; Arnouts 1996, A&amp;AS 117, 393).</li>
<li><b>Qt for Python</b> (PySide6), <b>pyqtgraph</b>, <b>matplotlib</b> (the PNG
figures), <b>NumPy</b>, <b>SciPy</b>, <b>Astropy</b>.</li>
<li>The example image of UGC 12517 (Hubble Space Telescope, WFC3/IR) and its mask come
from the elliprof package's example.</li>
<li>The command language follows MONSTA (Vista and Mongo) by John Tonry, Richard Stover
and Tod Lauer: the names and style are theirs; the commands are new code, written for
AstroVista Lite (see <i>Terminal syntax</i> and the <i>Command reference</i>).</li>
</ul>""",
    ),
    "elliprof": (
        "ELLIPROF: isophote fitting",
        """
<p>ELLIPROF fits ellipses to the isophotes of a galaxy, from the centre outwards, with the
<code>elliprof</code> package. Each ellipse gives the intensity, ellipticity, position
angle, centre and the 3θ and 4θ harmonics (boxy or discy shapes).</p>
<p><b>Centre (X0, Y0):</b> type it, <i>Pick on the image</i> (a click inside a detected
source takes the source's centre), double-click a source in the Sources table, or
or <i>Selected source</i>.</p>
<p><b>R0, R1, NR:</b> the first and last semi-major axes and the number of isophotes
between them; <i>Pick</i> sets a radius by clicking at that distance from the centre.
<b>NITER:</b> iterations of the fit. <b>RMSTAR:</b> leave out star-like bright samples
along each ellipse.</p>
<p><b>Sky:</b> a constant subtracted first; <i>Estimate</i> takes a clipped median of the
pixels the ticked masks leave.</p>
<p><b>Masks:</b> the <i>Mask with</i> list offers the mask layers and the buffers of the
image's size (a buffer masks where it is 0 or NaN); what is ticked (and a mask file) is
left out of the fit.
<i>Mask the sources</i> makes a layer of every detected source except the galaxy.</p>
<p><b>Results:</b> the profile table and plot, the isophotes on the image, and the model
and residual images as new buffers.</p>""",
    ),
    "sources": (
        "Source finder",
        """
<p>Finds the stars and galaxies in an image: everything that rises above the background
noise over enough connected pixels. For each source it measures the position, the shape
(an ellipse), the flux and magnitude, the size, and so on.</p>
<p><b>Settings</b> (SExtractor's names, in groups): <i>Detection</i>: the threshold, in
standard deviations of the background (RELATIVE) or in the image's units (ABSOLUTE), and
the minimum area. <i>Filtering</i>: smoothing before detection (the default kernel, or a
Gaussian of a given FWHM; matched or plain convolution). <i>Deblending</i>: how close
objects are split, and cleaning of false detections near bright ones.
<i>Background</i>: mapped from the image (mesh and filter sizes, filter threshold) or a
constant. <i>Photometry</i>: aperture diameters (MAG_APER, MAG_APER_2…), MAG_AUTO's Kron
factor and minimum radius, the light fraction of FLUX_RADIUS, the zero point and the gain.
<i>Masks</i>: mask layers whose pixels are left out.</p>
<p><b>After a run:</b> sources are drawn on the image (Kron ellipses); click one to select
it in the Sources table; plot any two columns in Plots; make masks of them in the Masks
menu (all, the selected one, those matching a condition such as
<code>FWHM_IMAGE &lt; 3</code>, or their segmentation).</p>
<p>The Source finder is built on SEP, which implements the algorithms of SExtractor; see
<i>Credits</i> for where they come from and what to cite. The settings keep SExtractor's
names, so its documentation explains them too.</p>""",
    ),
    "masks": (
        "Masks",
        """
<p>A mask layer marks pixels to leave out of a fit or a statistic. Any number of layers
can exist; the ticked ones in the ELLIPROF tab are used in the fit.</p>
<p><b>Drawing</b> (the tool bar): circle (click, or drag the radius), ellipse (drag from
the centre; the angle is set in the bar), box, polygon (click the corners; Enter or a
double-click closes it, Esc cancels, Backspace removes the last corner), brush, and the
eraser (a brush that unmasks).
<b>Alt</b> or <b>Shift</b> unmasks. The <b>Unmask</b> button (⊖, just before the circle
in the tool bar, and in the Masks menu) makes every mask tool unmask until it is clicked
again; with it on, Alt or Shift masks. A shape waiting to be applied is drawn dashed when
it will unmask.</p>
<p><b>Moving, resizing and rotating an ellipse, a box or a polygon.</b> Once drawn, the
shape waits as a magenta outline, not yet applied to the mask, with small squares and a
round handle above it. <b>Move</b> it by dragging it from inside. <b>Resize</b> it with
its squares: the ends of an ellipse's two axes (each axis grows or shrinks about the
centre), a box's corners (the opposite corner stays put), a polygon's corners (each moves
on its own). <b>Rotate</b> it with the round handle, about its centre (hold <b>Shift</b> for 15° steps), press <b>[</b> or <b>]</b> to turn
it by 5° (<b>{</b> <b>}</b> by 1°), or type the angle in the tool bar's <i>angle</i>
box. <b>Enter</b> or a double-click applies it; <b>Esc</b> cancels it. Drawing the next
shape, choosing another tool or running any command applies it too, so nothing drawn is
lost. A turned box is applied as a polygon of its four corners (<code>mask
polygon</code>); an ellipse keeps its angle (<code>mask ellipse ... ANGLE</code>).</p>
<p><b>Ctrl-click</b> (⌘-click on a Mac) removes the
masked patch under the cursor; holding Ctrl outlines it first.</p>
<p><b>Loading a mask:</b> right-click the Layers panel and choose <i>Load mask…</i>
(or Masks ▸ Load a mask…), or drop the file on the Layers panel: a FITS or legacy <code>.dmask</code> file,
whatever it is called. It becomes a mask layer, ticked for the fit, and a buffer
(1 = good, 0 = masked) that follows the layer's edits.</p>
<p><b>Making masks</b> (Masks menu): from the sources, from their segmentation, from
non-finite pixels or a range of values, or from a buffer.</p>
<p><b>Changing them</b> (right-click a layer): colour, opacity, grow, shrink, invert,
clear, fill, rename, copy, save, to a buffer, delete. A right-click anywhere on the
mask layers also offers <i>New mask…</i>, <i>Load mask…</i>, <i>Mask from a buffer…</i>,
<b>undo</b> (Ctrl+Z) and <b>redo</b> (Ctrl+Shift+Z), which cover the last 30 edits.</p>
<p><b>In AstroVista Lite a mask is always 0 for bad (masked) and 1 for good</b>, in
files, in buffers and in the fits; any value other than 0 counts as good, and NaN as
bad.</p>""",
    ),
    "layers": (
        "Layers",
        """
<p>What is drawn over the image: the isophotes, the sources, the ELLIPROF centre with R0
and R1, contours, the crosshair, and the mask layers. The compass and scale bar are always
shown, for images with a WCS. Tick to
show; click the colour box to change the colour; right-click a mask layer for what can be
done to it. The mask tools draw on the mask layer selected (in bold).</p>
<p>Any mask layer, and any image buffer of the same size (where it is 0 or NaN), can mask
the isophote fit: tick it in the ELLIPROF tab's Masks list.</p>""",
    ),
    "display": (
        "Display",
        """
<p>How the pixels are shown; nothing here changes their values.</p>
<p><b>Limits:</b> percentiles, ZScale, ZMax, min-max, typed, or dragged on the histogram
under the image. <b>Scale:</b> linear, log, power, square root, squared, asinh, sinh or
histogram equalisation. <b>Colours:</b> twelve colour maps, inverted if wanted.</p>
<p><b>Contrast and bias:</b> drag on the image with the right mouse button, or with Ctrl
(⌘ or Control on a Mac) held: across for the bias, up and down for the contrast.</p>
<p><b>Orientation:</b> North up (with a WCS), rotation and mirroring. <b>Contours</b> have
their own tool (the Contours tab). <b>Blink</b>: tick Blink above the image, choose the buffers in the menu beside it (its
first item, <i>Same scale</i>, shows them all with the same limits), click the image and
press ← and → to step between them. <b>Histogram</b> (above the image) shows
the histogram and the colour bar. <b>Go to</b> (Ctrl+G) a pixel or
an RA, Dec. The <b>crosshair</b> tool follows the mouse while you pan (drag) and zoom
(wheel) as usual; its lines stay upright on the screen; Alt with the arrow keys nudges it.
The Inspector (the first tab on the right) shows the pixel under it.</p>""",
    ),
    "contours": (
        "Contours",
        """
<p>Lines of equal brightness drawn over the image, with the settings of DS9's contour
dialog. They never change the pixels.</p>
<p><b>Levels:</b> how many, spread between the <i>low</i> and <i>high</i> limits according
to the <i>scale</i>: Linear spaces them evenly; Log, Square root or Asinh put more of them
at the faint end, where a galaxy's light fades; Histogram puts them at evenly spaced
fractions of the pixels. The limits come from min-max, ZScale, a percentile, or typed
values; <i>Same as the display</i> copies the image's display. <i>Generate</i> fills the
list of values, which can then be edited by hand.</p>
<p><b>Smoothing:</b> noise makes ragged contours. <i>Smooth</i> blurs the image with a
Gaussian of the smoothness (sigma, in pixels) first; <i>Block</i> averages blocks of that
many pixels (quicker on large images, coarser lines).</p>
<p><b>Lines:</b> colour, width and dashes. <i>Apply</i> draws them; <i>Clear</i> removes
them; the Contours layer shows or hides them.</p>""",
    ),
    "commands": (
        "The terminal",
        """
<p>Every action that changes data is a command, shown as it runs, so a session can be
read, repeated, saved (File ▸ Save the commands) and run again (<code>run file</code>).
<code>help</code> lists the commands; <code>help mask</code> the mask commands.
<i>Terminal syntax</i> explains how a line is written, and the <i>Command reference</i>
every command.</p>
<pre>rd 1 galaxy.fits[sci]
sources 1 DETECT_THRESH=2.5
mask new mymask 1
mask sources mymask WHERE='FWHM_IMAGE &lt; 3'
elliprof 1 X0=567 Y0=562 R0=9 R1=347 NR=23 MASK=mymask TOMODEL=3 TORESID=4
[4] = [1] - [2]
wd 3 residual.fits</pre>
<p><b>Image commands</b>, in the style of VISTA and MONSTA: <code>ac sc mc dc</code>
(a constant), <code>ai si mi di</code> (another buffer), <code>log ln sqrt inverse
round</code>, <code>mn max min abx print</code> (statistics), <code>shift rotate flip bin mash</code>,
<code>smooth filt zap clip skysub</code>, <code>fft noise gauss</code>, <code>ch
chead</code> (the header), <code>set typ</code> (variables, usable in any keyword:
<code>set xc=567</code> then <code>elliprof 1 X0=xc ...</code>). <code>help</code>
followed by a command explains it.</p>
<p><b>Text size:</b> View ▸ Terminal font size ▸ Larger (Ctrl+Alt+=), Smaller
(Ctrl+Alt+−) or Default size; the size is remembered.</p>
<p>Programs on the PATH run there too (<code>ls -l</code>, <code>ll</code>,
<code>grep</code>…); <code>!</code> forces the shell; Ctrl+C stops a program. Up and Down
go through the history. <b>Tab</b> completes what is being typed: commands, mask actions
and layer names, keywords such as <code>X0=</code> or <code>DETECT_THRESH=</code>, buffer
numbers and file names; when several fit, a list opens.</p>""",
    ),
}


class GuideWindow(QtWidgets.QDialog):
    """The documentation: a contents list with a search box, and the page chosen."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AstroVista Lite: documentation")
        self.resize(820, 580)
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search the documentation")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        left.addWidget(self.search)
        self.topics = QtWidgets.QListWidget()
        self.topics.setMaximumWidth(230)
        for key, (title, _text) in TOPICS.items():
            item = QtWidgets.QListWidgetItem(title.split(":")[0])
            item.setData(QtCore.Qt.UserRole, key)
            self.topics.addItem(item)
        left.addWidget(self.topics, 1)
        lay.addLayout(left)
        self.text = QtWidgets.QTextBrowser()
        self.text.setOpenLinks(False)  # topic: links change the page, others open outside
        self.text.anchorClicked.connect(self._link)
        lay.addWidget(self.text, 1)
        self.topics.currentItemChanged.connect(
            lambda item, _p: item and self.show_topic(item.data(QtCore.Qt.UserRole))
        )

    def _filter(self, text: str) -> None:
        """Only the pages containing every word typed."""
        words = text.lower().split()
        first = None
        for k in range(self.topics.count()):
            item = self.topics.item(k)
            title, page = TOPICS[item.data(QtCore.Qt.UserRole)]
            plain = re.sub("<[^>]+>", " ", title + " " + page).lower()
            hit = all(w in plain for w in words)
            item.setHidden(not hit)
            if hit and first is None:
                first = item
        if first is not None and words:
            self.topics.setCurrentItem(first)
            self.show_topic(first.data(QtCore.Qt.UserRole))
            for w in words:  # highlight the first place
                self.text.find(w)
                break

    def _link(self, url: QtCore.QUrl) -> None:
        """A link in a page: another page (topic:key), or a web page in the browser."""
        if url.scheme() == "topic":
            self.show_topic(url.path())
        else:
            from PySide6 import QtGui

            QtGui.QDesktopServices.openUrl(url)

    def show_topic(self, key: str) -> None:
        title, text = TOPICS[key]
        self.text.setHtml(f"<h3>{title}</h3>{text}")
        for k in range(self.topics.count()):
            if self.topics.item(k).data(QtCore.Qt.UserRole) == key:
                self.topics.blockSignals(True)
                self.topics.setCurrentRow(k)
                self.topics.blockSignals(False)


_window = None


def close_window() -> None:
    """Close the documentation window, if it is open (when the main window closes)."""
    global _window
    try:
        import shiboken6

        alive = _window is not None and shiboken6.isValid(_window)
    except ImportError:  # pragma: no cover
        alive = _window is not None
    if alive:
        _window.close()
        _window.deleteLater()
    _window = None


def show(key: str, parent=None) -> None:
    """Show the page ``key`` in the documentation window (one window, raised).  The window
    belongs to no other window, so it outlives whichever asked for it."""
    global _window
    try:
        import shiboken6

        alive = _window is not None and shiboken6.isValid(_window)
    except ImportError:  # pragma: no cover
        alive = _window is not None
    if not alive:
        _window = GuideWindow(None)
    _window.show_topic(key)
    _window.show()
    _window.raise_()
    _window.activateWindow()


def button(key: str, parent=None, label: str = "?") -> QtWidgets.QToolButton:
    """A small ? button that opens the topic ``key`` in the documentation (no hover text)."""
    b = QtWidgets.QToolButton(parent)
    b.setText(label)
    b.setProperty("help_topic", key)
    b.setAutoRaise(True)
    b.setStyleSheet("QToolButton { color: #8b9099; padding: 0 6px; }")
    b.clicked.connect(lambda: show(key, b.window()))
    return b


# ---- short explanations, shown by the small ? buttons beside the settings
TIPS = {
    # ELLIPROF
    "x0": "<b>X0, Y0: the centre of the galaxy</b>, in pixels. The ellipses are fitted "
    "around it. Use <i>Pick on the image</i> and click the bright "
    "core.",
    "r0": "<b>R0: the first (smallest) ellipse</b>, as its semi-major axis in pixels. Start "
    "a little outside the very centre, where the light changes too fast to fit: 3 to 10 "
    "pixels is typical.",
    "r1": "<b>R1: the last (largest) ellipse</b>, in pixels. Choose where the galaxy fades "
    "into the sky; going further only fits noise. <i>Pick</i> it by clicking there.",
    "nr": "<b>NR: how many ellipses</b> between R0 and R1. They are spaced so that each is "
    "a fixed factor larger than the one before. 15 to 30 is plenty.",
    "niter": "<b>NITER: how many times the fit is refined.</b> More iterations settle the "
    "ellipses better; 5 to 10 is usual.",
    "rmstar": "<b>RMSTAR</b> ignores bright, star-like points along each ellipse, so that "
    "a star on top of the galaxy does not pull the fit.",
    "sky": "<b>The sky</b> is the light of the background (the night sky, the detector), "
    "the same everywhere. It must be subtracted before measuring the galaxy. "
    "<i>Estimate</i> takes it from the pixels that are not masked.",
    "maskfile": "<b>A mask file</b> marks the pixels to leave out (other sources, "
    "bad pixels): 0 means masked, 1 means used. FITS or .dmask files work.",
    "maskwith": "<b>What masks the fit:</b> tick mask layers (drawn, loaded, or made from "
    "the sources) and image buffers (masked where they are 0). Masked pixels are left out.",
    "masksize": "<b>Size of the source masks</b>, in Kron ellipses (an ellipse holding "
    "most of each source's light). 1.5 covers the faint wings of most sources.",
    "results": "<b>The model</b> is the galaxy rebuilt from the fitted ellipses; <b>the "
    "residual</b> is the image minus the model, which shows what the ellipses miss "
    "(dust lanes, spiral arms, companions).  <b>Into</b> chooses the buffer each goes into: "
    "Auto takes a free one; the list offers the free buffers, and <i>Show the occupied "
    "buffers too</i> lets a result replace what is in a buffer.",
    # the pair fit
    "pair_sky": "<b>The sky by segmentation.</b> Every source found down to the threshold "
    "(0.5 sigma by default; lower is more conservative) is left out, grown by 2 pixels, and "
    "the sky is measured on what is left: its median (the histogram shows the pixels), the "
    "medians of cells over the frame, a plane through them (the gradient), and a verdict on "
    "how much sky there is. The verdict is advice: <i>Use this sky</i> takes the median as "
    "the base level whatever it says, and the verdict is kept with the fit. One sky is used "
    "for both galaxies and every cycle.",
    "pair_grow": "<b>The radius schedule.</b> Each galaxy is first fitted out to its "
    "<i>Initial R1</i>; then its R1 grows every cycle, by a factor (× 1.5), by a number of "
    "pixels (+ 20) or along a list you give (40, 60, 90, 130, 180), until it reaches the "
    "<i>Final R1</i>. A and B have their own schedules; the note under them shows the R1 of "
    "each fit. After both reach their final R1, the cycles go on there until they converge "
    "or the maximum is reached.",
    "pair_damp": "<b>Damping.</b> With a value below 1, what is subtracted from D when the "
    "companion is fitted is a blend: this share of its new model and the rest of its previous "
    "one (0.5 by default). On 12 simulated pairs it gave the most accurate profiles in half "
    "of them, most of all for a lopsided or a faint companion. Two undamped cycles always end a "
    "damped run, so the two profiles kept are ELLIPROF's own and the residual is D - sky - "
    "M_A - M_B of those models.",
    "pair_search": "<b>Search for the best fit</b> (on by default). Fits every combination of "
    "the damping values, the centre fitting and the growth factors, and keeps the stable fit "
    "with the smallest residual outside the cores (its RMS in units of the pixel noise; 1 is "
    "pure noise). Fixed centres are preferred: a fit with free centres is kept only when it "
    "lowers that residual by more than 0.01. A residual judges A + B together, and free "
    "centres can match it while moving light between A and B. On 24 simulated pairs this rule "
    "gave a smaller residual than simply keeping the smallest one, and more accurate profiles "
    "(median error 1.2 % against 1.7 %). Results > Pair > Search lists every variant and why "
    "one was kept.",
    "pair_converge": "<b>Convergence.</b> A cycle at the final radii counts as converged when "
    "A, B and A + B each changed by less than this fraction (their RMS change over the "
    "evaluation pixels). The run stops after this many converged cycles in a row (2 by "
    "default); 0 runs every cycle. The threshold is a starting value, not a validated "
    "scientific criterion.",
    "pair_onfail": "<b>When a fit fails.</b> A failed fit is never used: the galaxy keeps "
    "its last valid model. By default it then tries smaller steps: halfway, then a quarter "
    "of the way from its last valid radius to the one that failed, and goes on growing "
    "toward its final R1 in the next cycles; it stays put only when even a small step fails. "
    "It can instead stay at the radius of its last valid model, try the failed radius again "
    "in the next cycle, or the whole run can stop there. The iteration table "
    "and the report say which galaxy failed, at which radius and why.",
    "pair_block": "<b>Blocking a galaxy.</b> The region of this radius around this galaxy "
    "(shaped as its source, else a circle) is left out of the other galaxy's first fit, so "
    "that the other is first fitted alone. Block the smaller galaxy, out to where its light "
    "fades; after that, its model is taken out instead and only its core is left out. A "
    "block never covers the other galaxy's centre.",
    "pair_skyfactor": "<b>The sky factor.</b> The sky subtracted is factor x base. After a "
    "fit, the median m of the residual (image - sky - A - B) on the sky pixels suggests the "
    "next factor, factor + m / base: <i>Use the next factor</i>, then fit again, or let "
    "<i>Refine the factor</i> do it up to N times and keep the fit with the smallest |m|. "
    "ELLIPROF's models reach beyond R1, so their outer light is part of the residual.",
    "pair_centre": "<b>The centre</b> of this galaxy, in ELLIPROF's coordinates (the first "
    "pixel's centre is at 0.5). <i>Pick A</i> takes one click for A, then one for B. A click "
    "within 3 pixels of a found source takes the source's centre, and the card says so; "
    "Alt-click places the centre exactly where you click. The fit refines the centre of "
    "every ellipse (unless <i>Centres</i> says otherwise).",
    "pair_mode": "<b>Harmonics</b> (ELLIPROF's COS3X and COS4X): every isophote's fit measures "
    "the cos 3θ and cos 4θ terms; these choose what goes into the model. 2 puts in each "
    "isophote's own (ELLIPROF's default), 1 the median over all isophotes, 0 none. COS4X "
    "describes disky (positive) and boxy (negative) isophotes; COS3X lopsided ones. A "
    "negative COS3X fits the 6th order instead of the 3rd (-2 each, -1 median, -3 none); "
    "such a model is wrong beyond a turn of the position angle through 0/180 degrees, and a "
    "fit where elliprof warns of it fails. <i>Try</i> lists values to fit as well, on the fit "
    "kept (TRYCOS3X=, TRYCOS4X=): A's values with B as it is, then B's with A's best; one replaces "
    "the fit kept only when it lowers the residual by more than 0.01 of the noise.",
    "pair_runs": "<b>Runs to average</b> (RUNS=, JITTER=): the fit kept is fitted again, each time "
    "with every galaxy's R0 and R1 changed at random by two thirds of the percentage to the "
    "percentage, up or down, so that the fit may settle on a slightly different model. The "
    "runs are averaged: buffers get the mean A, B and A + B models, the mean residual (the "
    "image minus the mean sky minus the mean A + B), the spread of the models, and every "
    "run's own model and residual; the mean sky is in the report (a buffer too when the fits "
    "removed a sky tilt). A run whose "
    "residual is more than 50 % above the best is left out of the mean. The runs go "
    "<i>Parallel fits</i> at a time. SEED= repeats the same random changes.",
    "pair_joe": "<b>Joe's mode</b> (METHOD=JOE) follows Joe Jensen's way of fitting a pair "
    "by hand. Each galaxy's centre is left out (a circle of 0.2 of the separation, 6 to 30 px) "
    "while the other is fitted, because ELLIPROF goes astray on the residual of the other's "
    "centre. The smaller galaxy is fitted first, not far out, then the other on the image "
    "minus it; if the first cannot be started, the other starts. Then they alternate: each "
    "tries an R1 30 % larger and keeps it only as far as its outer ellipses, fitted with free "
    "centres, do not move towards the other galaxy. The model kept is fitted again with the "
    "galaxy's own centre setting. There is no search and no sky refinement, so it takes "
    "about a tenth of the elliprof calls of Automatic. On 18 simulated pairs its errors were "
    "close to Automatic's (the SBF amplitude in the residual within 3 % of the truth for "
    "half the galaxies, against 2 %; the model within 2.6 %, against 2.2 %), with worse "
    "worst cases; on three real fields it left 1.3 to 3 times more large-scale structure "
    "in the residual. Use it as a quick first look, and its steps to see where each galaxy "
    "is pulled. Its box in the panel sets which galaxy goes first, how much R1 grows each "
    "cycle, the centre left out, what counts as a pull (pixels, share of the radius, "
    "isophotes in a row, where the free-centre fit starts), the centres of the model kept, "
    "the cycles, the smoothing and the model beyond R1; <i>Joe's defaults</i> puts them "
    "back. Each galaxy's card sets its R0, its R1 (or automatic), its first R1 (or "
    "automatic), its NR (or automatic) and its harmonics (COS3X, COS4X).",
    "pair_shape": "<b>Outer ellipses</b> (Joe's mode, SHAPETOL=0.12): the outermost ellipses "
    "of a fit can turn much rounder or flatter than the rest, and the model, which ELLIPROF "
    "continues beyond R1 from them, then goes astray. At the end of Joe's mode each galaxy's "
    "3 outermost ellipses are compared with the 4 inside them (ellipticity and angle "
    "together, e exp(2i PA)), both with their median and with their trend in log radius; one "
    "off both by more than SHAPETOL is a bad ellipse. The model is also checked beyond R1: "
    "it must fall outwards and not be brighter than the light there. Then coarser isophotes "
    "(70 % of the automatic NR) and an R1 just inside the bad ellipse are tried, the largest "
    "fit that is clean is kept (with that R1 and NR from then on), and both galaxies are "
    "fitted once more. A change further in is left alone: it is the galaxy's own. On 18 "
    "simulated pairs it brought the SBF amplitude in the residual to within 1.6 % of the "
    "truth for half the galaxies (2.8 % without), and 6 fits ended stable instead of 1. "
    "0 turns it off.",
    "pair_pull": "<b>Pull check</b> (every complete pair fit; PULLCHECK=0 turns it off): each "
    "galaxy is fitted once more, with free centres, on the image minus the other's model and "
    "out to its R1. If its outer ellipses' centres move towards the other galaxy (by more "
    "than 0.5 px or 2 % of the radius, two isophotes in a row) the report says from which "
    "radius: beyond it the galaxy's light is not separated from the other's. This is Joe "
    "Jensen's test, done by eye in his pair fits. It does not change the result.",
    "pair_wings": "<b>Mask star wings and spikes</b> (WINGS=1): a first pass is fitted, its "
    "residual smoothed over 1.5 px and compared with its own local spread; light above 2.5 "
    "times that, joined to the mask of a bright source (at least 40 px), up to 150 px from "
    "it and never within 12 px of a galaxy's centre, is added to the mask. A star's wings and "
    "diffraction spikes reach beyond the source finder's segment and raise the SBF measured "
    "in the residual; specks of SBF or noise are not joined to a mask, so they stay.",
    "pair_parallel": "<b>Parallel fits</b> (PARALLEL=): the search fits each variant on its own "
    "(damping, centres, growth, harmonics); this many run at the same time, each ELLIPROF "
    "call a process of its own. The fit kept is the same whatever the number; only the time "
    "changes (on NGC 6768, 56 s one at a time, 24 s five at a time). When two or more fits "
    "are within 5 % of the best residual, their mean A + B model, mean residual (the image "
    "minus the mean sky minus the mean A + B), the spread of their models (pixel by pixel) "
    "and each fit's own model and residual are put in buffers as well: the spread shows "
    "where the answer depends on the choices searched.",
    "pair_fixctr": "<b>Centres</b>: <i>Free</i> lets every ellipse find its own centre (the "
    "default; a drift towards the other galaxy shows in the centre track); <i>Fixed</i> keeps "
    "all at X0, Y0; <i>Median</i> sets all to the median of the fitted centres after each "
    "iteration.",
    "pair_core": "<b>The core of this galaxy</b> is left out while the <i>other</i> galaxy is "
    "fitted, where this galaxy's model is least reliable. <i>Auto</i>: an ellipse of the "
    "source found here (its Kron ellipse, at most 0.65 of the distance to the other centre "
    "and 50 pixels), or a circle when no source matches. It never covers the other centre, "
    "and it is fixed for the whole run.",
    "pair_adaptive": "<b>Adaptive outer radius</b>: when the outer ellipses fail their checks "
    "(crossing, a jump, an impossible value) beyond 0.65 R1, the galaxy is fitted again to a "
    "smaller R1 (at most 3 times, never below half the R1 asked for). The radius asked for, "
    "the last one tried and the one used are all reported.",
    "pair_masks": "<b>Fitting masks</b>: the layers left out of every fit of A and of B. Only "
    "layers of role <i>fit</i>, or without a role, are listed: a sky layer covers the "
    "galaxies themselves and is never used in a fit.",
    "pair_auto": "<b>Settings</b>. <i>Automatic</i> (the default): you choose the sky, the "
    "masks and the two centres. <i>Automatic, adjustable</i>: every setting shown at the "
    "same values, to change any of them; the radii and the isophotes stay automatic until "
    "their auto boxes are unticked. <i>Manual</i>: every setting, the radii typed or "
    "picked. In the automatic settings the rest is "
    "what worked best on simulated pairs, among them pairs of ellipticals with surface "
    "brightness fluctuations, globular clusters, stars and photon noise. R1 is where each "
    "galaxy's light, on the side away from the other, falls below the noise sigma, measured "
    "again on the image minus the other's model after a first pass; the first fits use a "
    "third of R1 (at most 0.35 of the separation, at least 9 px) and grow by 1.5; isophotes "
    "about 30 % apart; centres fixed; the parameters smoothed over 3 isophotes (TIE=-3); "
    "bright outliers along each isophote rejected (RMSTAR); the model beyond R1 continued by "
    "a Sérsic law fitted to the outer profile; the sky factor refined from the residual sky "
    "3 times (with a measured sky); 8 variants of damping, centres and growth fitted and the "
    "best kept by the residual. The report and the export record every value used. <i>Joe's mode</i>: Joe Jensen's way of fitting a pair, quick and without a search (see Joe's mode under Every setting explained). Every complete fit ends with a pull check of the outer ellipses (see Pull check).",
    "pair_tie": "<b>Smoothing with radius</b> (TIE): after each iteration, the ellipticity, "
    "angle and centre are smoothed over neighbouring isophotes: -3 averages over 3 "
    "(recommended), -5 over 5; -1 turns it off; k >= 0 fits a polynomial of order k in "
    "r^1/4. On simulated pairs with SBF it halved the isophote-to-isophote jitter of the "
    "shapes, which shows as a four-leaved pattern in the residual; over more isophotes it "
    "smooths away real twists.",
    "pair_rmstar": "<b>RMSTAR</b>: along each isophote, samples brighter than the median + 4 "
    "x (upper quartile - median) are ignored in that iteration. Globular clusters and faint "
    "stars on a galaxy are part of its detection segment, so <i>Mask the other objects</i> "
    "leaves them; RMSTAR keeps them from pulling the isophotes. On simulated pairs it "
    "brought the SBF amplitude in the residual to within 4 % of the truth for 9 galaxies in "
    "10.",
    "pair_skyplane": "<b>Remove a sky gradient</b> (SKYPLANE=1): the pair is fitted once, a "
    "plane is fitted to its residual (the medians of 32-pixel boxes of unmasked pixels), and "
    "when the gradient is at least 3 times its error the image minus the plane's tilt is "
    "fitted. The level at the frame's centre stays the sky you chose. Elliptical models "
    "cannot take up a gradient, so the residual shows it even on frames a halo fills, where "
    "the sky alone cannot be measured; on NGC 6768 it found 84 counts per 1000 px. The "
    "report gives the plane.",
    "pair_tail": "<b>Model beyond R1</b> (TAIL): elliprof extends its model to the frame's "
    "corners with ln I linear in r^1/4 through the last two isophotes: right for a de "
    "Vaucouleurs profile, too bright for a shallower one or when the last two are noisy. In "
    "a pair that light is subtracted from the companion's image and leaves dark regions. "
    "<i>Measured</i> (recommended, the automatic settings) takes the light measured in rings "
    "of the last isophote's shape on the image minus the companion, never rising outwards, "
    "and a Sérsic law beyond the last ring; <i>Sérsic</i> continues the last isophote by a "
    "Sérsic law fitted to the outer 60 % of the profile, on the last isophote's ellipse; "
    "<i>exponential</i> by the outer slope; <i>cut off</i> tapers it to zero; <i>elliprof</i> "
    "keeps elliprof's.",
    "pair_r1auto": "<b>R1</b>, the final outer radius. <b>auto</b> (also for the initial R1 "
    "and NR): where the median light in "
    "3 px rings, on the half facing away from the other galaxy and on unmasked pixels, falls "
    "below the noise sigma (at most 0.9 of the distance to the frame's edge). Untick it to type "
    "or pick a radius.",
    "pair_objects": "<b>Mask the other objects</b> makes the mask layer <i>objects</i> (role "
    "fit) and ticks it: the pixels of every source the Source finder detects, grown by this "
    "many pixels, except the sources at A's and B's centres (and the segment under each "
    "centre). It finds the sources first if this image has no source-finder run, with the "
    "Source finder's settings; a lower threshold there masks more of the faint objects. The "
    "galaxies are never masked from each other. Edit the layer with the mask tools, or "
    "untick it to fit without it. Command: mask segments objects X0=xa,xb Y0=ya,yb GROW=2.",
    "pair_cycles": "<b>Cycles</b>: after the start (A, then B on the image minus A), each "
    "cycle fits A on the image minus B, then B on the image minus the new A. 4 cycles make "
    "10 elliprof calls. More cycles do not guarantee a stable result.",
    # the Source finder
    "THRESH_TYPE": "<b>How the threshold is given:</b> RELATIVE, in standard deviations "
    "of the background noise (3 means three times the noise); ABSOLUTE, in the image's own "
    "units.",
    "DETECT_THRESH": "<b>The detection threshold:</b> how much brighter than the "
    "background a pixel must be to count. Lower finds fainter sources, but also more "
    "noise. 1.5 to 5 is usual.",
    "DETECT_MINAREA": "<b>The minimum area:</b> how many connected bright pixels make a "
    "source. Larger values ignore hot pixels and tiny noise peaks.",
    "FILTER": "<b>Filter:</b> smooth the image slightly before looking for sources. It "
    "helps find faint sources the size of a star.",
    "FILTER_FWHM": "<b>Filter size:</b> the width (FWHM, in pixels) of the smoothing. "
    "About the size of a star works best; 0 uses a small default kernel.",
    "FILTER_TYPE": "<b>Filter type:</b> matched (recommended) takes the noise into "
    "account; conv is a plain smoothing.",
    "DEBLEND_NTHRESH": "<b>Deblending levels:</b> how finely touching sources are "
    "examined to split them into separate objects. 32 is the usual value.",
    "DEBLEND_MINCONT": "<b>Deblending contrast:</b> how bright a part must be (as a "
    "fraction of the whole) to become a separate object. Smaller values split more; "
    "0.005 is usual, 1 never splits.",
    "CLEAN": "<b>Clean:</b> remove false detections in the halos of bright objects.",
    "CLEAN_PARAM": "<b>Cleaning efficiency:</b> larger values clean more aggressively. "
    "1 is usual.",
    "BACK_TYPE": "<b>The background:</b> AUTO measures it across the image (it may vary "
    "from place to place); MANUAL uses one value everywhere.",
    "BACK_VALUE": "<b>Background value</b> (for MANUAL): the level subtracted everywhere, "
    "in the image's units.",
    "BACK_SIZE": "<b>Background mesh:</b> the background is measured in squares of this "
    "many pixels. Make it larger than the sources (and much larger than stars), but small "
    "enough to follow real changes. 32 to 256.",
    "BACK_FILTERSIZE": "<b>Background filter:</b> the background map is smoothed over "
    "this many squares, so that one bright source does not spoil it. 3 is usual.",
    "BACK_FILTTHRESH": "<b>Filter threshold:</b> only squares that differ from their "
    "neighbours by more than this are smoothed. 0 smooths them all.",
    "PHOT_APERTURES": "<b>Apertures:</b> diameters (pixels) of circles in which the light "
    "of every source is added up, giving MAG_APER, MAG_APER_2... Separate several with "
    "commas: 5,10,20.",
    "PHOT_AUTOPARAMS": "<b>MAG_AUTO's ellipse:</b> the Kron factor (2.5 holds about 94% "
    "of the light) and the smallest size allowed (3.5), separated by a comma.",
    "PHOT_FLUXFRAC": "<b>Flux fraction:</b> FLUX_RADIUS is the radius holding this "
    "fraction of the light: 0.5 gives the half-light radius.",
    "MAG_ZEROPOINT": "<b>Zero point:</b> turns counts into magnitudes: magnitude = zero "
    "point - 2.5 log10(counts). Find it in the image's documentation (0 gives "
    "instrumental magnitudes).",
    "GAIN": "<b>Gain:</b> electrons per count, used for the errors of the fluxes. 0 "
    "leaves out the sources' own noise.",
    "MASK": "<b>Leave out:</b> names of mask layers whose pixels are ignored when finding "
    "sources (bad columns, a bright star's halo...).",
    # display
    "limits": "<b>Limits:</b> the pixel values shown as darkest and brightest. Percentages "
    "clip the brightest and faintest pixels; ZScale suits faint detail. Type values to set "
    "them yourself.",
    "scale": "<b>Scale:</b> how values between the limits become brightness. Log and Asinh "
    "show faint outskirts and bright centres together; Linear is true to the numbers.",
    "colours": "<b>Colours:</b> the colour map. Grey is faithful; colour maps help the eye "
    "see small differences. Invert swaps dark and bright.",
    "histogram": "<b>Histogram:</b> how many pixels have each value, with the limits in "
    "yellow (drag them), and the colour bar.",
    "blink": "<b>Blink:</b> tick it, choose two or more buffers in the menu, click the "
    "image, then press ← and → to flip between them and spot the differences. Keep "
    "<i>Same scale</i> on, so that equal brightness looks equal. Zoomed in, it keeps the "
    "place and zoom: on the sky when the WCS match, otherwise on the image.",
    "orientation": "<b>Orientation:</b> North up turns the image so that North is up and "
    "East left, as on the sky. Rotating and mirroring change only the display.",
    "contrast": "<b>Contrast and bias</b> change the colour map only: contrast its "
    "steepness, bias where its middle lies. Or drag on the image with the right button.",
    # contours
    "c_levels": "<b>Levels:</b> how many contour lines.",
    "c_scale": "<b>Scale:</b> how the lines are spread between low and high: Linear evenly, "
    "Log more of them at the faint end (good for galaxies).",
    "c_limits": "<b>Limits:</b> the faintest and brightest contour, from the image's "
    "statistics or typed (User).",
    "c_values": "<b>Values:</b> the contour levels, one per line. <i>Generate</i> fills "
    "them; you can edit them by hand.",
    "c_smoothing": "<b>Smoothing:</b> noisy images give ragged contours. Smooth blurs "
    "first (a Gaussian of this many pixels); Block averages blocks of pixels.",
    "c_lines": "<b>Lines:</b> the colour, width and style of the contour lines.",
}

# what the columns of the sources table mean
COLUMNS = {
    "#": "the number of the source",
    "x": "X_IMAGE: the centre's column (the first pixel's centre is 1)",
    "y": "Y_IMAGE: the centre's row",
    "mag": "MAG_AUTO: the magnitude, from the light in the Kron ellipse (smaller is brighter)",
    "± mag": "the uncertainty of the magnitude",
    "flux": "FLUX_AUTO: the light in the Kron ellipse, in the image's units",
    "r½": "FLUX_RADIUS: the radius holding half the light, pixels",
    "FWHM": "FWHM_IMAGE: the width of the source, pixels (stars: the seeing)",
    "ellip": "ELLIPTICITY: 1 - b/a; 0 for round sources",
    "θ": "THETA_IMAGE: the angle of the long axis, degrees from the x axis",
    "flags": "FLAGS: warnings (1: close neighbours, 2: blended, 4: saturated, 8: at an edge)",
    "RA": "ALPHA_J2000: right ascension, degrees",
    "Dec": "DELTA_J2000: declination, degrees",
}


def tip_button(key: str, parent=None) -> QtWidgets.QToolButton:
    """A small ? beside a setting: a click opens the documentation at the setting's entry
    (Every setting explained); there is no hover text."""
    b = QtWidgets.QToolButton(parent)
    b.setText("?")
    b.setAutoRaise(True)
    b.setFixedSize(18, 18)
    b.setStyleSheet("QToolButton { color: #8b9099; border: none; }")
    b.setFocusPolicy(QtCore.Qt.NoFocus)
    b.setProperty("help_setting", key)
    b.clicked.connect(lambda: show_setting(key))
    MADE_TIPS.add(key)
    return b


MADE_TIPS: set = set()  # the settings a ? was made for (the tests check them)


def tip_group(key: str) -> tuple:
    """(heading, topic) of the tool a setting belongs to."""
    if key.startswith("c_"):
        return "Contours", "contours"
    if key in ("limits", "scale", "colours", "histogram", "blink", "orientation", "contrast"):
        return "Display", "display"
    if key.upper() == key:  # the Source finder's settings are in SExtractor's names
        return "Source finder", "sources"
    return "ELLIPROF", "elliprof"


def _settings_page() -> str:
    """Every setting a ? explains, grouped by tool, each with an anchor of its own."""
    parts = []
    for group in ("ELLIPROF", "Source finder", "Display", "Contours"):
        keys = [k for k in TIPS if tip_group(k)[0] == group]
        topic = tip_group(keys[0])[1]
        parts.append(
            f"<h4>{group}</h4><p style='color:#8b9099'>The whole story: "
            f"<a href='topic:{topic}'>{TOPICS[topic][0]}</a>.</p>"
        )
        for k in keys:  # the Source finder's keyword first, as typed in the terminal
            keyword = f"<code>{k}</code>: " if group == "Source finder" else ""
            parts.append(f"<p><a name='{k}'></a>{keyword}{TIPS[k]}</p>")
    rows = "".join(f"<tr><td><b>{c}</b></td><td>{t}</td></tr>" for c, t in COLUMNS.items())
    parts.append(f"<h4>The columns of the Sources table</h4><table cellspacing='3'>{rows}</table>")
    return (
        "<p>What each setting does, all in one place: a click on the small ? beside a "
        "setting opens this page at its entry.</p>" + "".join(parts)
    )


def show_setting(key: str) -> None:
    """The documentation, at the entry of one setting."""
    show("settings")
    _window.text.scrollToAnchor(key)
    # a window opened just now lays its page out later: scroll again once it has
    text = _window.text
    QtCore.QTimer.singleShot(0, lambda: text.scrollToAnchor(key))


def with_tip(widget: QtWidgets.QWidget, key: str) -> QtWidgets.QWidget:
    """``widget`` with a ? beside it explaining ``key``."""
    box = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(widget, 1)
    row.addWidget(tip_button(key))
    if key in TIPS and not widget.toolTip():
        import re

        widget.setToolTip(re.sub("<[^>]+>", "", TIPS[key]))
    return box


TOPICS["faq"] = (
    "FAQ",
    """
<p><b>Where does it run?</b> On macOS 13 or later (Apple silicon and Intel) and on Linux
with glibc 2.34 or later (Ubuntu 22.04+, Debian 12+, Fedora 35+, RHEL 9+; on ARM, Ubuntu
24.04+), with Python 3.10 to 3.13 (3.14 needs a C compiler for SEP). It is on PyPI:
<code>pip install AstroVista</code>. With an older Python, or none, a conda environment
works: <code>conda create -n astrovista -c conda-forge python=3.12 pip</code>, then
<code>conda activate astrovista</code> and <code>pip install AstroVista</code>.</p>
<p><b>Do I need to install anything besides AstroVista Lite?</b> No: <code>pip install
AstroVista</code> brings everything, the Source finder included.</p>
<p><b>Linux says it could not load the Qt platform plugin "xcb"</b>: a minimal system (a
server, WSL, a container) lacks a few libraries every desktop has. On Debian or Ubuntu:
<code>sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libegl1 libgl1 libfontconfig1
libdbus-1-3</code>; on Fedora: <code>sudo dnf install xcb-util-cursor libxkbcommon-x11
mesa-libEGL mesa-libGL fontconfig dbus-libs</code>.</p>
<p><b>Which images can it open?</b> FITS images, whatever the file is called, compressed or
not, one image of a multi-extension file, and legacy .dmask masks.</p>
<p><b>Is my data changed?</b> Never on disk: results go to new buffers, and you save what
you want with File ▸ Save buffer. Display settings never change pixel values.</p>
<p><b>What do 0 and 1 mean in a mask?</b> 0 is bad (masked), 1 is good.</p>
<p><b>What is a buffer?</b> A numbered slot holding one image; the list on the left shows
them all, with how each was made.</p>
<p><b>How do I repeat what I did?</b> Every action is a command: File ▸ Save the commands,
then File ▸ Run a command file.</p>
<p><b>Why NaN in my result?</b> NaN marks pixels without a value: masked, outside the
image, or undefined (x/0). Statistics and mean/median/sum leave them out.</p>
<p><b>How do I cite it?</b> Cite its GitHub repository, <a
href="https://github.com/ekourkchi/AstroVista-Lite">github.com/ekourkchi/AstroVista-Lite</a> (see <i>Credits</i>), and for the measurements elliprof
(isophotes) and SEP and SExtractor (the Source finder).</p>
<p><b>Is there more?</b> AstroVista Lite is the lite edition of AstroVista, an image
analysis studio in development.</p>""",
)


TOPICS["allcommands"] = (
    "Command reference",
    "<p>Every command the terminal knows, with its syntax, its keywords and an example "
    "(the examples run as written). <code>help name</code> explains one in the terminal "
    "and Tab completes it; <i>Terminal syntax</i> explains how a line is written. Commands "
    "marked <i>MONSTA name</i> share their name with a command of MONSTA, John Tonry's "
    "image analysis system; their options are AstroVista Lite's own, as given here.</p>"
    + reference.reference_html(),
)
TOPICS["terminal"] = ("Terminal syntax", reference.TERMINAL_HTML)

TOPICS["settings"] = ("Every setting explained", _settings_page())

# Help ▸ Tips, one at a time
TIP_LIST = [
    "Click a source on the image to see its row in Results ▸ Sources; double-click a row "
    "to take it as ELLIPROF's centre.",
    "Drag on the image with the right mouse button to change the contrast and brightness; "
    "Reset puts them back.",
    'Add <code>tag="a name"</code> to any command that makes an image, to name it.',
    "Press Tab in the terminal to complete commands, keywords and file names.",
    "Ctrl-click (⌘-click) a masked patch to remove it; Ctrl+Z undoes.",
    "Tick Blink, choose buffers, click the image and press ← → to compare them.",
    "Results ▸ Profile ▸ R¼ law fits de Vaucouleurs' law and gives the effective radius.",
    "<code>[2] = [1] - median([1])</code> subtracts a constant sky in one line.",
    "Click a folder in an <code>ls</code> listing to go into it, a FITS file to open it.",
    "File ▸ Save the commands keeps your session; File ▸ Run a command file replays it.",
    "The ? beside a setting explains it; Help ▸ Documentation (F1) explains everything.",
    "<code>fitsinfo file.fits</code> lists the images inside a file; "
    "<code>rd 1 file.fits[sci]</code> opens one.",
]

TOPICS["pair"] = (
    "Fitting a pair of galaxies",
    """
<p>When two galaxies overlap, each one's light lies on the other's ellipses. Masking the
neighbour is not enough: its halo reaches far beyond any mask, and pulls the ellipticity,
the position angle and the profile of the galaxy being fitted towards it. The <b>Pair fit</b>
tool fits both, in turn, each with the other's model taken out.</p>

<h3>1. The sky, first</h3>
<p><i>Measure sky</i> finds every source down to the threshold (0.5 sigma by default; a lower
threshold masks more, which is conservative), leaves them out, grown by 2 pixels, and
measures the sky on what is left: the median (the histogram shows those pixels), the medians
of cells across the frame, a plane through the cells (the gradient, as an angle on the
image from +x, not a position angle on the sky), an approximate sampling error and the
spread between cells. These are kept apart: neither is the total error of the sky.</p>
<p>The verdict says how much sky there is: <b>usable_support</b>,
<b>candidate_inspect_support</b> (with the reasons: too few pixels, cells only along one
edge or a line, cells that disagree, a gradient larger than the noise) or
<b>no_estimate</b>. The verdict is advice: <i>Use this sky</i> takes the median as the
base level whatever it says, and the verdict is kept with the fit. Enough sky is needed,
but it is not proof: a halo too faint to detect can remain in the "sky". Leaving out
everything above 0.5 sigma also reads pure noise a little low (about 0.2 sigma). Large
galaxies often fill an image; you can also type a value (from a larger image, an earlier
measurement…), with a note on where it comes from. Changing the image or the masks makes a
measurement stale.</p>
<p>The sky layer it makes (<code>skymask</code>, role <i>sky</i>) covers both galaxies, so
it is never used in a fit. <i>Show the sky mask</i> draws it on the image.</p>
<p><b>The sky factor.</b> The sky subtracted is <i>factor × base</i>. After a fit, step 5
shows the residual (image − sky − A − B) on the sky pixels, with its histogram: its median
m suggests the next factor, factor + m / base (<i>Use the next factor</i>, then fit again).
<i>Refine the factor</i> does this up to N times and keeps the fit with the smallest |m|;
Results ▸ Pair ▸ Sky lists every fit. ELLIPROF's models reach beyond R1, so their outer
light is part of that residual: the factor makes the sky and the models' halos together
match the sky pixels.</p>

<h3>2. The two galaxies</h3>
<p><i>Pick A</i>, click A's centre, then B's. Give each its own R0 and R1 (semi-major
axes, in pixels), number of ellipses, harmonics and centre fitting. The dashed circles show
R0 and R1; they are guides, not fits.</p>

<h3>3. The method</h3>
<pre>start:    A on the image            (B's core left out)
          B on the image - A        (A's core left out)
cycle k:  A on the image - B, then B on the image - A    (k = 1 ... CYCLES)
result:   A, B, A + B, and the residual: image - sky - A - B, on every pixel</pre>
<p>Every call subtracts the same sky once. A fit counts only when it passes every check
(the right number of ellipses, finite values, ellipticity below 0.95, ellipses of one
galaxy that do not cross each other, no lone jump of the centre, the axis ratio or the
angle, the last ellipse at R1). Ellipses of A and B may cross: they belong to different
galaxies. A rejected fit is never used: the galaxy keeps its last good model. NITER counts
the iterations inside each elliprof call; cycles count the turns of A and B. Four cycles
make 10 elliprof calls.</p>
<p><b>The iterative decomposition</b> (the default). A, the larger galaxy, is fitted first
on the original image D out to its <i>Initial R1</i>; then B on D − M_A; then A on D − M_B
and B on D − M_A, cycle after cycle, each R1 growing by its own schedule (× a factor, + a
number of pixels, or a list) to its <i>Final R1</i>. Every input is made afresh from D,
which is never changed. The galaxies are not masked from each other: the companion's
current model is subtracted instead. Centres are fixed by default. At the final radii the
cycles go on until A, B and A + B change by less than the threshold for two cycles in a
row, or the maximum is reached. A failed fit is never used: the galaxy keeps its last
valid model and stays at its radius, tries again, or the run stops (your choice). The
iteration table lists every call as it runs; double-click a row to see its model and the
image it was fitted on (<code>pairstep</code>). <i>Copy report</i> puts a plain-text report
on the clipboard.</p>
<p>When a galaxy's first fit fails (its neighbour is too bright), it is started from a
smaller R1 (0.7, 0.5 or 0.35 of it): such a <i>bootstrap</i> model only starts the cycles
and is never part of a result.</p>

<h3>4. How it ended</h3>
<ul>
<li><b>stable_valid_pair</b>: both fits passed in the last cycle, and A, B and A + B each
changed by less than 2 %.</li>
<li><b>valid_pair_not_stable</b>: both passed, but something still changes (the reason says
what; a stable sum with changing parts is not stable).</li>
<li><b>retained_after_failed_update</b>: a fit failed in the last cycle; the result is the
last cycle in which both passed (a change of zero there means nothing).</li>
<li><b>incomplete_initialization</b>: no cycle in which both passed.</li>
<li><b>cancelled</b>: stopped; the result is the last cycle in which both passed.</li>
</ul>
<p>Results ▸ Pair plots the two profiles and the change per cycle. <i>Blink the result</i>
steps through image − sky, A + B, image − sky − B and image − sky − A on one scale. Look at
the residual: rings, a bowl, a bright patch next to a dark one, and centre tracks drifting
towards the other galaxy are signs of a poor separation. Passing the checks and a small
change do not prove a unique separation, nor that the residual is ready for SBF.</p>

<h3>From the terminal</h3>
<pre>sky 1 SEGMENT
pair 1 SKY=1000 MASK=stars \\
       A: NAME=A X0=294.3 Y0=263.8 R0=3 R1=120 \\
       B: NAME=B X0=421.18 Y0=322.97 R0=3 R1=50 NR=20
pairexport 1 my_pair</pre>
<p>Shared settings come before <code>A:</code>; each galaxy's after its label.
<code>help pair</code> lists them all; <code>SKYFACTOR=</code> and <code>SKYITER=</code> are
the sky factor. File ▸ Open the pair example loads a synthetic pair whose true answer is in
its FITS header. File ▸ Open a pair folder opens a folder holding a FITS image, its
<code>.dmask</code> masks and a DS9 region file with the two centres (points labelled
<code>A:</code> and <code>B:</code>), with starting radii to check.</p>
""",
)

TOPICS["skyplane"] = (
    "The sky as a plane",
    """
<p>The sky is not always flat: scattered light, a bright star beyond the frame or a poor
flat field make one side of the image brighter than the other. A single sky level is then
too high on one side and too low on the other, and the outer isophotes of a galaxy lean
towards the bright side. The residual of a fit shows it as a broad tilt.</p>
<p><b>Tools ▸ Sky plane</b> fits the sky as a plane: a level at the centre of the frame and
a gradient along x and y, each with its error. The sources are found as for <i>Measure
sky</i>, but on the image minus the plane's tilt, so that a bright side is not taken for a
source; the plane is fitted to the medians of boxes of what is left (32 pixels by default,
weighted by their sky pixels, clipped at 3 MAD), and the two steps are repeated until the
tilt changes by less than 0.02 sigma.</p>
<p><b>Reading the result.</b> A gradient of 3 times its error or more is significant. The
boxes' scatter says whether the sky is a plane: about 1 is what the noise gives; much more
means light that is not sky, such as a halo filling the frame, and the gradient may then be
the halo's. The small map shows what is left on the flat image, by cells.</p>
<p><b>Using it.</b> <i>Add the flat image</i> (ticked by default) makes the image minus the
tilt, its level kept, and measures its sky. Open the flat image and fit it: in the Pair fit,
<i>Use this sky</i> takes the flat image's measurement, and the sky factor works as
before. The plane itself and the image minus the whole plane can be added too.</p>
<p>In the terminal: <code>skyplane 1 TOFLAT=2</code>; <code>help skyplane</code> lists the
rest.</p>
""",
)

# the topics in reading order (the guide window and the Help menu follow it)
ORDER = (
    "overview lesson recipes files images display blinking layers masks elliprof profile pair skyplane "
    "sources arithmetic tags imagecommands contours inspect commands terminal listing saving keys "
    "glossary settings faq problems allcommands credits"
).split()
TOPICS = {key: TOPICS[key] for key in ORDER + [k for k in TOPICS if k not in ORDER]}
