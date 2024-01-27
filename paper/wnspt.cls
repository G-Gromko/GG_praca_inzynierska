% Klasa dokument\'ow do sk{\l}adu prac magisterskich/licencjackich
% na wydziale Zarz\k{a}dzania Uniwersytetu Gda\'nskiego ( wersja 1.06)
%
% This work may be distributed under the terms of the LaTeX Project
% Public License, as described in lppl.txt in the base LaTeX
% distribution.  Either version 1.0 or, at your option, any later version.
%
% Klasa wmpmgr mo\.ze by\'c rozpowszechniana na zasadach publicznej
% licencji LaTeX Project (The LaTeX project public license, LPPL,
% por. http://www.latex-project.org/lppl/)
% Istota LPPL sprowadza si\k{e} do tego, \.ze modyfikacje pliku dozwolone
% s\k{a} wylacznie pod warunkiem zmiany jego nazwy.
%
% Zmiany na potrzeby WMP AJD by (c) A. Zbrzezny, 2012
% Zmiany na potrzeby WMP UJD by (c) A. Zbrzezny, 2019
% Zmiany na potrzeby WNSPT UJD by (c) A. Zbrzezny, 2021
%
% Copyright (c) T. Przechlewski (rudolf.von.ems@gmail.com), 2002--2004, 2005, 2008, 2009, 2011
%
% Poprawki (bez polskich znakow zeby sie dawalo czytac w roznych systemach):
% --------
% wersja: 1.11 | 19/06/2009:
%    Usunięto zadawniony błąd polegający na używaniu \input wewnątrz \DeclareOption
%    przy okazji dodano opje tgtermes oraz matematykę w krojach tgtermes/tgpagella
%
% wersja: 1.10 | 19/06/2009:
%     Slowa kluczowe+streszczenie sa opcjonalne. Jezeli nie sa podane nie sa drukowane
%
% wersja: 1.06 | 02/03/2009:
%     Dodano alternatywny sposob formatowania literatury adekwatny w przypadku
%     uzywania stylu autor-rok. Dodana opcja `breaklinks' do wywolania pakietu hyperref
%     [zakladamy, ze nikt juz nie uzywa formatow dvi/ps]
%
% wersja: 1.05: 
%     Zmieniono palatino na tgpagella (bo palatino.sty zostal
%     zastapiony przez tgpagella.sty)
%
% wersja: 1.04 | 07/06/2007: 
%     Dodano polecenie \UniversityName
%
% wersja: 1.03 | 18/06/2005: 
%     Usunieto kilka istotnych bledow zgloszonych przez Piotra Tylskiego,
%     w tym: brak wpisow `Bibliografia' i Oswiadczenie w spisie tresci
%
% wersja: 1.02 | 07/4/2005: 
%     Dodane polecenie `\oswiadczenie' wstawiajace tekst deklaracji okreslonej
%     w pismie okolnym Rektora UG nr 4/R/04 (z dnia 13.12.2004)
%
% wersja: 1.01 | 23/4/2004: 
%     Poprawiony blad w slowie Zrodlo
%
%
% Dzi\k{e}kuj\k{e} Marcinowi Woli\'nskiemu za zgod\k{e} na wykorzystanie 
% fragment\'ow jego klasy opracowanej w tym samym celu co niniejsza
% dla Wydzia\{l}u Matematyki, Informatyki i Mechaniki, UW.
%

\def\ThisClassVersionId{v1.10}
%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{wnspt}[2009/06/19 \ThisClassVersionId, Praca magisterska/licencjacka]

\def\@baseclass{report}
\def\@rodzajpracy{in\.zynierska}
\def\@BInterlinia{\relax}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\@baseclass}}
\PassOptionsToClass{a4paper,twoside,openany,11pt}{\@baseclass}
% %% >> zmiana 2011/1/28 >>
\newif\if@wnsptauthoryear  \@wnsptauthoryearfalse
\newif\if@wnsptWithTgPagella \@wnsptWithTgPagellafalse
\newif\if@wnsptWithTgTermes \@wnsptWithTgTermesfalse
\newif\if@wnsptDratfMode \@wnsptDratfModefalse
%
\DeclareOption{autorrok}{\@wnsptauthoryeartrue}
\newif\if@wnsptbackref \DeclareOption{backref}{\@wnsptbackreftrue}
\DeclareOption{licencjacka}{\def\@rodzajpracy{licencjacka}}
\DeclareOption{skorowidz}{\input{makeidx.sty}\makeindex}
\DeclareOption{tgpagella}{\@wnsptWithTgPagellatrue}
\DeclareOption{palatino}{\@wnsptWithTgPagellatrue}
\DeclareOption{tgtermes}{\@wnsptWithTgTermestrue}
\DeclareOption{times}{\@wnsptWithTgTermestrue}
% opcja `brudnopis' jest poprawiona -- teraz wykorzystuje pakiet `prelim2e'
\DeclareOption{brudnopis}{\@wnsptDratfModetrue }
\DeclareOption{xodstep}{\AtEndOfClass{\linespread{1.3}}}
\ProcessOptions
%
\LoadClass{\@baseclass}
%
\if@wnsptWithTgPagella \RequirePackage{tgpagella,pxfonts,qpxmath}\RequirePackage[T1]{fontenc} \fi
\if@wnsptWithTgTermes \RequirePackage{tgtermes,txfonts,qtxmath}\RequirePackage[T1]{fontenc} \fi
%
\if@wnsptDratfMode \RequirePackage{prelim2e}
\renewcommand{\PrelimWords}{\ThisDocVersionId}\linespread{1.0}
\fi
%
\usepackage{graphicx}
\usepackage{polski}
%%
\if@wnsptbackref 
  \usepackage[breaklinks,pagebackref]{hyperref}\else
  \usepackage[breaklinks]{hyperref} \fi
  \if@wnsptauthoryear 
  \typeout{*** Cytowanie autor-rok ***}\RequirePackage{natbib}\fi
% %% << zmiana 2011/1/28 <<
\textwidth\paperwidth
\advance\textwidth -55mm
\oddsidemargin-1in
\advance\oddsidemargin 30mm
\evensidemargin-1in
\advance\evensidemargin 25mm
\topmargin -1in
\advance\topmargin 25mm
\setlength\textheight{48\baselineskip}
\addtolength\textheight{\topskip}
\marginparwidth15mm
%
\widowpenalty999999
\tolerance450
\pretolerance250
\hfuzz=1.5pt
\hbadness1450
%
\def\AndNextAuthor{\ifnum\language=0\& \else i \fi}
\newcount\authornumber
\authornumber=0
%
\def\author#1{%
  \global\advance\authornumber\@ne
  \expandafter\def\csname theauthor\number\authornumber\endcsname
      {\ignorespaces#1\unskip}%
  \expandafter\def\csname theauthorid\number\authornumber
  \endcsname{???\ClassError{wnspt}{Brak numeru albumu}\@ehc}%
  \expandafter\let\csname thenetaddress\number\authornumber\endcsname\relax }
%
\def\nralbumu#1{%
  \expandafter\def\csname theauthorid\number\authornumber\endcsname
      {\ignorespaces#1\unskip}}
\def\email#1{% -- opcjonalny --
  \expandafter\def\csname thenetaddress\number\authornumber\endcsname
      {\ignorespaces#1\unskip}}

%%\def\authorlist#1{\def\@author{#1}}
\def\@author{\@defaultauthorlist}

\newcount\count@@

\def\@defaultauthorlist{%
   \count@=\authornumber
   \count@@=0
   \loop
   \ifnum\count@>0
      \advance\count@@ by 1
      \begingroup \Large\bf
      \ignorespaces\csname theauthor\number\count@@\endcsname\par
      %\normalsize \rm nr~albumu:
      \rm nr~albumu:
      \ignorespaces\csname theauthorid\number\count@@\endcsname\par
      % adres email nie jest drukowany -- ale si{\e} przyda
      %%\ignorespaces\csname thenetaddress\number\count@@\endcsname
      \par 
      \medskip \endgroup
      \advance\count@ by -1
   \repeat
  }

 % <<
\def\UniversityName#1{\def\Univ@Name{#1}}
\def\Univ@Name{UNIWERSYTET~JANA~D{\L}UGOSZA~w~CZ\k{E}STOCHOWIE}%
 % <<
\renewcommand\maketitle{%
  \begin{titlepage}%
    \let\footnotesize\small
    \let\footnoterule\relax
    \let \footnote \thanks
    \vspace{2mm minus 30mm}
    \begin{center}%
      {\Large \bfseries{\textbf{\Univ@Name}}\par}
      \par 
    \end{center}%
     %\vspace{0.0cm plus 1fill}
     \vspace{0.2cm}
    \begin{center}%
      \includegraphics[scale=0.8]{logo_ujd.png}
    \end{center}%
    \begin{center}%
      {\LARGE \textbf{Wydzia{\l} Nauk Ścis{\l}ych, Przyrodniczych i~Technicznych}\par}
      \par 
    \end{center}%
    \vspace{6mm}
    \begin{flushleft}
      {\large \textsf{Kierunek: \@kierunek}}\\[0pt]
      {\large \textsf{Specjalno{\'s}{\'c}: \@specjalnosc}\par}
    \end{flushleft}%
    \vspace{6mm}
    \begin{center}%
      {\huge \@author\par}
    \end{center}%
     \vspace{6mm plus 1mm minus 2mm}
    \begin{center}%
      {\LARGE\textbf{\@title}\par}
      \vspace{1.8cm plus 1.5fill}
      \begin{flushright}\large
        \begin{tabular}{l}
          \textsf{\Large Praca \@rodzajpracy{} przygotowana}\\[1pt]
          \textsf{\Large pod kierunkiem}\\[1pt]
          \textsf{\Large \@opiekun}\\[1pt]
          \textsf{\Large \@opiekunpom}
        \end{tabular}
      \end{flushright}
      \vspace{10mm plus .1fill}
      \textsf{\Large \@miejsce\space \@date}\par
    \end{center}
    \@thanks
  \end{titlepage}%
  %
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
  %
  % Jezeli podano slowa kluczowe wydrukuj
  \clearpage 
  \thispagestyle{empty}
  \ifvoid\abspagebox \@latex@warning{*** Nie podano streszczenia ***}%
  \else \InsertAbstractHere \fi
  \ifx\@keywords\EmptyKwsList \@latex@warning{*** Nie podano slow kluczowych ***}%
  \else \section*{S{\l}owa kluczowe}
   \@keywords
  \fi
  %\cleardoublepage %%
  \tableofcontents
  %\listoftables
  %\listoffigures
} %% --renewcommand: maketitle --

%%\def\nralbumu#1{\gdef\@nralbumu{#1}}
%%\def\@nralbumu{???\ClassError{wnspt}{Brak numeru albumu}\@ehc}
\def\kierunek#1{\gdef\@kierunek{#1}}
\def\@kierunek{???\ClassError{wnspt}{Nie podano kierunku studiow}\@ehc}
\def\specjalnosc#1{\gdef\@specjalnosc{#1}}
\def\@specjalnosc{???\ClassError{wnspt}{Nie podano specjalnosci}\@ehc}
\def\instytut#1{\gdef\@instytut{#1}}
\def\@instytut{???\ClassError{wnspt}{Nie podano instytutu}\@ehc}
\def\opiekun#1{\gdef\@opiekun{#1}}
\def\opiekunpom#1{\gdef\@opiekunpom{#1}}
\def\@opiekun{???\ClassError{wnspt}{Brak danych opiekuna pracy}\@ehc}
\def\miejsce#1{\gdef\@miejsce{#1}}
\def\@miejsce{Cz\k{e}stochowa}
%
\def\keywords#1{\gdef\@keywords{#1}}
% slowa kluczowe sa opcjonalne:
%%\def\@keywords{???\ClassError{wnspt}{Brak slow kluczowych}\@ehc}
\def\@keywords{????}
\def\EmptyKwsList{????}
%%
\def\klasyfikacja#1{\gdef\@klasyfikacja{#1}}
\def\@klasyfikacja{???\ClassError{wnspt}{Brak klasyfikacji
    tematycznej}\@ehc}
%
\newbox \abspagebox %% *tp **
\def\InsertAbstractHere{\unvbox \abspagebox}
%
\renewenvironment{abstract}{\global 
  \setbox \abspagebox \vbox \bgroup 
  \section*{Streszczenie}  }
 {\egroup}
%
\renewcommand*\@seccntformat[1]{\csname the#1\endcsname.\enspace}
\def\numberline#1{\hb@xt@\@tempdima{#1.\hfil}}
\renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\mdseries
      \leaders\hbox{$\m@th
        \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
      \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
%
\def\@makechapterhead#1{%
  %\vspace*{50\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
        \normalsize \bfseries \MakeUppercase{\@chapapp}\space \thechapter
        \par\nobreak
        \vskip\baselineskip 
    \fi
    \interlinepenalty\@M
    \LARGE \bfseries #1\par\nobreak
    \vskip 40\p@
  }}
%
\def\@makeschapterhead#1{%
  %\vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \LARGE \bfseries  #1\par\nobreak
    \vskip 40\p@
  }}
%
\renewcommand{\section}{\@startsection{section}{1}{\z@}%
    {-3.5ex \@plus -1ex \@minus -.2ex}%
    {2.3ex \@plus.2ex}%
    {\reset@font\Large\bfseries\raggedright}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
    {-3.25ex\@plus -1ex \@minus -.2ex}%
    {1.5ex \@plus .2ex}%
    {\reset@font\large\bfseries\raggedright}}
\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{\z@}%
    {-3.25ex\@plus -1ex \@minus -.2ex}%
    {1.5ex \@plus .2ex}%
    {\reset@font\normalsize\bfseries\raggedright}}
\renewcommand{\paragraph}{???\ClassError{wnspt}{Zbyt duzo srodtytulow}\@ehc}%
%
% Skorowidz: dodanie kresek w pozycjach skorowidza stopnia drugiego
\renewenvironment{theindex}{%
   \clearpage \refstepcounter{chapter}% potrzebne do prawid{\l}owego
     % dzia{\l}ania zak{\l}adek w dokumencie PDF
   \columnseprule \z@ \columnsep 35\p@
   \twocolumn[\@makeschapterhead{\indexname}]%
   \@mkboth{\indexname}{\indexname}%
   \addcontentsline{toc}{chapter}{\indexname}% 
   \thispagestyle{plain}\parindent\z@
   \parskip\z@ \@plus .3\p@\relax
   \let\item\@idxitem }
   {\clearpage }
\def\indexemdashes{%
\def\@idxitem{\par\hangindent 0pt}
\def\subitem{\par\hangindent 0pt --- }
\def\subsubitem{\par\hangindent 0pt --- --- }
\def\indexspace{\par \vskip 10pt plus 5pt minus 3pt\relax}}
\indexemdashes
%
% Wprowadzenie/zako\'nczenie
\newcommand{\introduction}{%\doublepage
   \chapter*{\IntroName\@mkboth{\IntroName}{\IntroName}}%
\addcontentsline{toc}{chapter}{\IntroName}}
%
\newcommand{\summary}{%\doublepage
   \chapter*{\SummaryName\@mkboth{\SummaryName}{\SummaryName}}%
\addcontentsline{toc}{chapter}{\SummaryName}}
% O\'swiadczenie (dodane 7.4.2005)
\def\oswiadczenieText{%
Ja, ni\.zej podpisany(a) o\'swiadczam, i\.z przed{\l}o\.zona praca 
dyplomowa zosta{\l}a wykonana przeze mnie samodzielnie, 
nie narusza praw autorskich, interes\'ow prawnych 
i~materialnych innych os\'ob.\endgraf }
%
\long\def\oswiadczenie{%%\newpage 
\if@twocolumn \onecolumn \fi
%%\centerline{\large\textbf{O\'swiadczenie}}
%%\vspace*{50mm}
\chapter*{O\'swiadczenie}%
\addcontentsline{toc}{chapter}{O\'swiadczenie}%
\par 
\vskip6mm
\noindent
\oswiadczenieText
\par\vskip12mm 
\centerline{\vbox{%
\hbox to.8\textwidth{\hbox to45mm{\dotfill}\hss 
   \hbox to45mm{\dotfill}} 
\hbox to.8\textwidth{\hbox to45mm{\hss data\hss}\hss
\hbox to45mm{\hss podpis\hss}}%
}}\par
\newpage }
%
% Polecenia listoffigures/listoftables uproszczono oraz 
% dodano \addcontentsline, tak aby tytu{\l} spisu pojawi{\l} 
% si\k{e} w spisie tre\'sci i zak\ladkach w pliku PDF:
\renewcommand\listoffigures{%
    \chapter*{\listfigurename
      \@mkboth{\MakeUppercase\listfigurename}%
              {\MakeUppercase\listfigurename}}%
    \addcontentsline{toc}{chapter}{\listfigurename}
    \@starttoc{lof}}
%
\renewcommand\listoftables{%
    \chapter*{\listtablename
      \@mkboth{%
          \MakeUppercase\listtablename}{\MakeUppercase\listtablename}}%
    \@starttoc{lot}%
    \addcontentsline{toc}{chapter}{\listtablename}}
%
\renewenvironment{thebibliography}[1]
     {\chapter*{\bibname}%
      \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
      \addcontentsline{toc}{chapter}{\bibname}%<--
      \list{%%
          \@biblabel{\@arabic\c@enumiv}}%\fi}%
           {%%
             \settowidth\labelwidth{\@biblabel{#1}}%
             \leftmargin\labelwidth
             \advance\leftmargin\labelsep
             \@openbib@code
             \usecounter{enumiv}%
             \let\p@enumiv\@empty
             \renewcommand\theenumiv{\@arabic\c@enumiv}}%
           \sloppy
           \clubpenalty4000
           \@clubpenalty \clubpenalty
           \widowpenalty4000%
           \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}

% Uproszczono polecenie \@makecaption, tak \.ze podpisy s\k{a}
% teraz sk{\l}adane od lewego marginesu. 
% Dodano polecenie \source do sk{\l}adania \'xr\'od{\l}a
\long\def\@makecaption#1#2{\par
  \vskip\abovecaptionskip
  \begin{flushleft}\small\textbf{#1}. #2\end{flushleft}
  %\vskip\belowcaptionskip % niepotrzebne bo po caption 
     %winno by\'c \'xr\'od{\l}o
  \par
}
\def\source#1{\par \begin{flushleft}\small 
\SourceName:~#1\end{flushleft}}
%
% Oznakowanie wersji dokumentu na ka\.zdej stronie w postaci
% adresu e-mail pierwszego autora (je\.zeli jest zdefiniowany) lub imienia
% i nazwiska pierwszego autora + tego co autor wpisa{\l} jako
% argument polecenia \nrwersji + bie{\.z}\k{a}ca data.
\def\nrwersji#1{\def\ThisDocVersion{#1}}\nrwersji{????}%
\def\ThisDocVersionId{%
  \expandafter \ifx \csname theauthor1 \endcsname \undefined   
   \else \csname theauthor1\endcsname \fi
   %% adres moze zawierac nie-litery (np `_' powoduje blad), pomijam
   %\expandafter \ifx \csname thenetaddress1\endcsname \undefined   
   %\else \csname thenetaddress1\endcsname \fi 
  \space --\space wersja \ThisDocVersion\space z\space 
  \number\day/\number\month/\number\year~r. } 
%
\def\IntroName{Wst\k{e}p}
\def\SummaryName{Zako\'nczenie}
\def\SourceName{\'Zr\'od{\l}o}
%
\endinput
