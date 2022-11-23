#!/usr/bin/perl

use strict;
use utf8;

my @gold_spaces = (1);
my $gold_wc = 0;
my $pointer = 0;

open(GOLD, "<:utf8", $ARGV[0]) or die;
while (<GOLD>) {
    s/\r?\n$//;
    foreach my $word (split /\s+/, $_) {
        $pointer += length($word);
        $gold_spaces[$pointer] = 1;
        $gold_wc++;
    }
}

my @test_spaces = (1);
my $test_wc = 0;
$pointer = 0;

open(TEST, "<:utf8", $ARGV[1]) or die;
while (<TEST>) {
    s/\r?\n$//;
    foreach my $word (split /\s+/, $_) {
        $pointer += length($word);
        $test_spaces[$pointer] = 1;
        $test_wc++;
    }
}

if ($#test_spaces != $#gold_spaces) {
   print "WARNING: Different sizes of test and gold files: TEST: $#test_spaces, GOLD: $#gold_spaces\n";
}

my $begin_ok = 0;
my $correct_count = 0;
foreach my $i (0 .. $#gold_spaces) {
    if ($gold_spaces[$i] == 1 && $test_spaces[$i] == 1) {
        $correct_count++ if $begin_ok;
        $begin_ok = 1;
    }
    elsif ($gold_spaces[$i] != $test_spaces[$i]) {
        $begin_ok = 0;
    }
}

my $precision = $correct_count / $test_wc;
my $recall = $correct_count / $gold_wc;
my $f1 = 2 * $precision * $recall / ($precision + $recall);

printf("P:%.3f, R:%.3f, F:%.3f\n", $precision, $recall, $f1);

